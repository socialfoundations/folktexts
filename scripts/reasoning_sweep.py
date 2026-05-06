"""Reasoning-mode-only sweep on a small representative subset.

Reasoning on transformers takes 2-3 hours per cell at the standard
subsampling=0.01 — dominating the wall-time budget. This harness runs
a focused reasoning subset at subsampling=0.005 (~1.6k rows) so the full
sweep fits in ~5-6 hours overnight.

Scope:
- `Qwen--Qwen3-4B-Thinking-2507` × (reasoning, reasoning_think) — primary target.
- `meta-llama--Meta-Llama-3-8B-Instruct` × reasoning — secondary (regular
  instruct model running CoT, no thinking-template).
- both backends, seed 42.

Usage:
    python scripts/reasoning_sweep.py
    python scripts/reasoning_sweep.py --subsampling 0.005
    python scripts/reasoning_sweep.py --backends vllm
    python scripts/reasoning_sweep.py --report-only
"""
from __future__ import annotations

import argparse
import gc
import json
import logging
import sys
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import torch  # noqa: E402

from folktexts.benchmark import Benchmark, BenchmarkConfig  # noqa: E402
from folktexts.cli._utils import get_or_create_results_dir  # noqa: E402
from folktexts.llm_utils import load_model_tokenizer  # noqa: E402

# ---------- Constants ---------------------------------------------------------

MODELS_ROOT = Path("/fast/groups/sf/huggingface-models")
DATA_DIR = Path("/fast/acruz/data/folktables")
RESULTS_DIR = REPO_ROOT / "results" / "reasoning-sweep"
LOGS_DIR = RESULTS_DIR / "logs"
TASK = "ACSIncome"
DEFAULT_SUBSAMPLING = 0.005   # ~1.6k rows (half of standard) to bound wall time
DEFAULT_SEED = 42
BATCH_SIZE = 16
DEFAULT_BACKENDS = ["transformers", "vllm"]


@dataclass(frozen=True)
class Spec:
    model_name: str
    use_thinking: bool       # ReasoningQA enable_thinking flag

    @property
    def slug(self) -> str:
        t = "_thinking" if self.use_thinking else ""
        return f"{self.model_name}__reasoning{t}"


# Default scope: focus on the model where reasoning matters most.
DEFAULT_SPECS: list[Spec] = [
    Spec("Qwen--Qwen3-4B-Thinking-2507", use_thinking=True),
    Spec("Qwen--Qwen3-4B-Thinking-2507", use_thinking=False),
    Spec("meta-llama--Meta-Llama-3-8B-Instruct", use_thinking=False),
]


@dataclass(frozen=True)
class Cell:
    spec: Spec
    backend: str

    @property
    def slug(self) -> str:
        return f"{self.spec.slug}__{self.backend}"


def _result_dir_for(cell: Cell) -> Path:
    return RESULTS_DIR / cell.backend / f"model-{cell.spec.model_name}_task-{TASK}"


def _find_existing(cell: Cell, subsampling: float) -> Optional[Path]:
    parent = _result_dir_for(cell)
    if not parent.exists():
        return None
    for path in parent.glob("**/results.bench-*.json"):
        try:
            with open(path) as f:
                d = json.load(f)
        except Exception:
            continue
        cfg = d.get("config", {})
        if (
            cfg.get("reasoning_prompting") is True
            and cfg.get("enable_thinking") == cell.spec.use_thinking
            and cfg.get("seed") == DEFAULT_SEED
            and cfg.get("batch_size") == BATCH_SIZE
            and abs((cfg.get("dataset_subsampling") or 0) - subsampling) < 1e-6
        ):
            return path
    return None


def run_cell(cell: Cell, subsampling: float, *, skip_existing: bool = True) -> tuple[dict | None, Optional[Exception]]:
    existing = _find_existing(cell, subsampling) if skip_existing else None
    if existing:
        with open(existing) as f:
            d = json.load(f)
        return ({
            "roc_auc": float(d["roc_auc"]),
            "ece": float(d["ece"]),
            "brier_score_loss": float(d["brier_score_loss"]),
            "n_samples": int(d["n_samples"]),
            "results_path": str(existing),
        }, None)

    model_path = MODELS_ROOT / cell.spec.model_name
    print(f"\n=== {cell.slug}  loading {model_path} ===", flush=True)
    try:
        if cell.backend == "vllm":
            from folktexts.llm_utils import load_vllm_model
            # Reasoning needs ample output budget; ACS prompts are short, so 8k
            # max_model_len is plenty.
            model, tokenizer = load_vllm_model(
                model_path.as_posix(),
                max_model_len=8192,
                seed=DEFAULT_SEED,
            )
        else:
            model, tokenizer = load_model_tokenizer(model_path.as_posix())

        config = BenchmarkConfig(
            reasoning_prompting=True,
            enable_thinking=cell.spec.use_thinking,
            batch_size=BATCH_SIZE,
            seed=DEFAULT_SEED,
        )
        bench = Benchmark.make_acs_benchmark(
            task_name=TASK,
            model=model,
            tokenizer=tokenizer,
            data_dir=DATA_DIR.as_posix(),
            config=config,
            subsampling=subsampling,
            backend=cell.backend,
            model_name_or_path=model_path.as_posix() if cell.backend == "vllm" else None,
        )
        base_results_dir = _result_dir_for(cell).parent
        results_dir = get_or_create_results_dir(
            model_name=cell.spec.model_name,
            task_name=TASK,
            results_root_dir=base_results_dir.as_posix(),
        )
        bench.run(results_root_dir=results_dir, fit_threshold=False)
        res = bench.results
        metrics = {
            "roc_auc": float(res["roc_auc"]),
            "ece": float(res["ece"]),
            "brier_score_loss": float(res["brier_score_loss"]),
            "n_samples": int(res["n_samples"]),
            "results_path": str(Path(res["results_dir"]) / f"results.bench-{res['benchmark_hash']}.json"),
        }
        print(f"<<< {cell.slug}: AUC={metrics['roc_auc']:.4f} ECE={metrics['ece']:.4f}", flush=True)
        del bench, model, tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return (metrics, None)
    except Exception as exc:
        tb = traceback.format_exc()
        log_path = LOGS_DIR / f"{cell.slug}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text(f"Failed cell: {cell.slug}\n\n{tb}\n")
        print(f"!! {cell.slug} FAILED ({type(exc).__name__}): {exc}", flush=True)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return (None, exc)


def write_report(specs: list[Spec], backends: list[str], subsampling: float) -> None:
    out_path = RESULTS_DIR / "REPORT.md"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    GATE_AUC = 0.02
    GATE_ECE = 0.03

    lines: list[str] = []
    lines.append("# Reasoning sweep — vLLM vs transformers")
    lines.append("")
    lines.append(
        f"Reasoning-mode validation on a focused subset, "
        f"subsampling={subsampling} (~{int(subsampling*332901)} rows). "
        f"`Δ = vllm − transformers`. Cells with `|ΔAUC|>{GATE_AUC}` or "
        f"`|ΔECE|>{GATE_ECE}` are **bolded**. Tolerance is wider than "
        f"non-reasoning gates because reasoning at temperature=0 is still "
        f"sensitive to attention-kernel divergences across long generations."
    )
    lines.append("")
    lines.append("| Model | thinking | tf AUC | vl AUC | ΔAUC | tf ECE | vl ECE | ΔECE | tf 0.5-rate | vl 0.5-rate |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|")
    for spec in specs:
        tf_metrics: dict | None = None
        vl_metrics: dict | None = None
        for backend in backends:
            cell = Cell(spec, backend)
            existing = _find_existing(cell, subsampling)
            if not existing:
                continue
            with open(existing) as f:
                d = json.load(f)
            metrics = {
                "roc_auc": float(d["roc_auc"]),
                "ece": float(d["ece"]),
            }
            csv_path = d.get("predictions_path")
            if csv_path and Path(csv_path).exists():
                import pandas as pd
                df = pd.read_csv(csv_path)
                metrics["fail_rate"] = float(((df["risk_score"] - 0.5).abs() < 1e-12).mean())
            else:
                metrics["fail_rate"] = float("nan")
            if backend == "transformers":
                tf_metrics = metrics
            else:
                vl_metrics = metrics

        thinking_tag = "✓" if spec.use_thinking else "—"

        def fmt(v: float | None) -> str:
            return f"{v:.3f}" if v is not None else "—"

        def delta(a: float | None, b: float | None, gate: float) -> str:
            if a is None or b is None:
                return "—"
            d = a - b
            cell = f"{d:+.3f}"
            return f"**{cell}**" if abs(d) > gate else cell

        tf_auc = (tf_metrics or {}).get("roc_auc")
        vl_auc = (vl_metrics or {}).get("roc_auc")
        tf_ece = (tf_metrics or {}).get("ece")
        vl_ece = (vl_metrics or {}).get("ece")
        tf_fail = (tf_metrics or {}).get("fail_rate")
        vl_fail = (vl_metrics or {}).get("fail_rate")
        lines.append(
            f"| `{spec.model_name}` | {thinking_tag} "
            f"| {fmt(tf_auc)} | {fmt(vl_auc)} | {delta(vl_auc, tf_auc, GATE_AUC)} "
            f"| {fmt(tf_ece)} | {fmt(vl_ece)} | {delta(vl_ece, tf_ece, GATE_ECE)} "
            f"| {fmt(tf_fail)} | {fmt(vl_fail)} |"
        )

    out_path.write_text("\n".join(lines) + "\n")
    print(f"Wrote -> {out_path}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--backends", default=None,
                        help=f"Comma-separated subset of {DEFAULT_BACKENDS}.")
    parser.add_argument("--subsampling", type=float, default=DEFAULT_SUBSAMPLING,
                        help=f"Default {DEFAULT_SUBSAMPLING} (~1.6k rows; half of standard).")
    parser.add_argument("--no-skip", action="store_true",
                        help="Re-run cells even if a matching JSON exists.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--report-only", action="store_true")
    args = parser.parse_args()

    backends = (
        [b.strip() for b in args.backends.split(",") if b.strip()]
        if args.backends else list(DEFAULT_BACKENDS)
    )
    unknown = [b for b in backends if b not in DEFAULT_BACKENDS]
    if unknown:
        raise SystemExit(f"Unknown backend(s): {unknown}. Known: {DEFAULT_BACKENDS}")

    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(levelname)s %(message)s")

    specs = list(DEFAULT_SPECS)
    cells = [Cell(s, b) for b in backends for s in specs]

    if args.report_only:
        write_report(specs, backends, args.subsampling)
        return 0

    print(f"Planned cells ({len(cells)}):")
    for c in cells:
        marker = "  [SKIP — existing]" if (not args.no_skip and _find_existing(c, args.subsampling)) else ""
        print(f"  - {c.slug}{marker}")
    if args.dry_run:
        return 0

    fails: list[str] = []
    for c in cells:
        metrics, exc = run_cell(c, args.subsampling, skip_existing=not args.no_skip)
        if exc is not None:
            fails.append(c.slug)

    write_report(specs, backends, args.subsampling)
    if fails:
        print(f"\n{len(fails)} cells failed: {fails}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
