"""Modern + thinking-model coverage sweep.

The Table 1 sweep covers the paper's 2024 model list. This harness extends
coverage to gemma-3, Qwen3, and Qwen3-Thinking checkpoints — modern
architectures that exercise paths that didn't exist when the paper was
written (especially `enable_thinking`, MoE routing, and longer context).

Per (model, mode) pair, the harness runs both backends and writes a
side-by-side comparison report.

Usage:
    python scripts/extended_sweep.py                       # default tier (small modern models)
    python scripts/extended_sweep.py --tier all            # everything (multi-day on 1× GPU)
    python scripts/extended_sweep.py --models gemma-3-4b-it,Qwen3-4B
    python scripts/extended_sweep.py --modes baseline,chat_mcq
    python scripts/extended_sweep.py --backends vllm
    python scripts/extended_sweep.py --report-only
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
from typing import Iterable, Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import torch  # noqa: E402

from folktexts.benchmark import Benchmark, BenchmarkConfig  # noqa: E402
from folktexts.cli._utils import get_or_create_results_dir  # noqa: E402
from folktexts.llm_utils import load_model_tokenizer  # noqa: E402

# ---------- Constants ---------------------------------------------------------

MODELS_ROOT = Path("/fast/groups/sf/huggingface-models")
DATA_DIR = Path("/fast/acruz/data/folktables")
RESULTS_DIR = REPO_ROOT / "results" / "extended-sweep"
LOGS_DIR = RESULTS_DIR / "logs"
TASK = "ACSIncome"
SUBSAMPLING = 0.01      # ~3.3k rows
SEED = 42
BATCH_SIZE = 16
DEFAULT_BACKENDS = ["transformers", "vllm"]


# Mode tag -> (numeric, use_chat, reasoning, enable_thinking).
MODE_FLAGS: dict[str, tuple[bool, bool, bool, bool]] = {
    "baseline":        (False, False, False, False),  # zero-shot MCQ, no chat
    "chat_mcq":        (False, True,  False, False),
    "chat_numeric":    (True,  True,  False, False),
    "reasoning":       (False, False, True,  False),  # CoT on, thinking off
    "reasoning_think": (False, False, True,  True),   # CoT on, thinking on (chat-template uses enable_thinking=True)
}

# Per-tier model lists. `chat_modes` controls which mode is run for which model.
# Keys are model dirnames under /fast/groups/sf/huggingface-models/.
@dataclass(frozen=True)
class ModelSpec:
    name: str
    modes: tuple[str, ...]
    notes: str = ""


# ---- Model menus -------------------------------------------------------------

# Base (non-instruct) models — only the baseline mode makes sense; chat/reasoning
# without instruct training is noise.
BASE_MODELS: list[ModelSpec] = []

# Modern instruct models. Tier 1 = small, tier 2 = mid, tier 3 = large.
TIER1_MODELS: list[ModelSpec] = [
    ModelSpec("google--gemma-3-1b-it",      ("baseline", "chat_mcq", "chat_numeric", "reasoning")),
    ModelSpec("google--gemma-3-4b-it",      ("baseline", "chat_mcq", "chat_numeric", "reasoning")),
    ModelSpec("Qwen--Qwen3-1.7B",           ("baseline", "chat_mcq", "chat_numeric", "reasoning")),
    ModelSpec("Qwen--Qwen3-4B",             ("baseline", "chat_mcq", "chat_numeric", "reasoning")),
    ModelSpec("Qwen--Qwen3-4B-Instruct-2507", ("baseline", "chat_mcq", "chat_numeric", "reasoning")),
    ModelSpec("Qwen--Qwen3-4B-Thinking-2507", ("chat_mcq", "chat_numeric", "reasoning", "reasoning_think")),
]

TIER2_MODELS: list[ModelSpec] = [
    ModelSpec("google--gemma-3-12b-it",     ("chat_mcq", "chat_numeric", "reasoning")),
    ModelSpec("Qwen--Qwen3-14B",            ("chat_mcq", "chat_numeric", "reasoning")),
]

TIER3_MODELS: list[ModelSpec] = [
    ModelSpec("google--gemma-3-27b-it",     ("chat_mcq", "chat_numeric", "reasoning")),
    ModelSpec("Qwen--Qwen3-32B",            ("chat_mcq", "chat_numeric", "reasoning")),
    ModelSpec(
        "Qwen--Qwen3-30B-A3B-Thinking-2507",
        ("chat_mcq", "chat_numeric", "reasoning", "reasoning_think"),
        notes="MoE 30B/A3B; reasoning trace expensive",
    ),
]

TIERS: dict[str, list[ModelSpec]] = {
    "tier1": TIER1_MODELS,
    "tier2": TIER1_MODELS + TIER2_MODELS,
    "tier3": TIER1_MODELS + TIER2_MODELS + TIER3_MODELS,
    "all":   TIER1_MODELS + TIER2_MODELS + TIER3_MODELS,
}


@dataclass(frozen=True)
class Cell:
    model_name: str
    mode: str
    backend: str

    @property
    def slug(self) -> str:
        return f"{self.model_name}__{self.mode}__{self.backend}"


# ---------- Idempotency ------------------------------------------------------

def _result_dir_for(cell: Cell) -> Path:
    return RESULTS_DIR / cell.backend / f"model-{cell.model_name}_task-{TASK}"


def _find_existing(cell: Cell) -> Optional[Path]:
    flags = MODE_FLAGS[cell.mode]
    numeric, use_chat, reasoning, thinking = flags
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
            cfg.get("numeric_risk_prompting") == numeric
            and cfg.get("use_chat_template") == use_chat
            and cfg.get("reasoning_prompting") == reasoning
            and cfg.get("enable_thinking") == thinking
            and cfg.get("seed") == SEED
            and cfg.get("batch_size") == BATCH_SIZE
            and abs((cfg.get("dataset_subsampling") or 0) - SUBSAMPLING) < 1e-6
        ):
            return path
    return None


def _load_metrics(cell: Cell) -> Optional[dict]:
    p = _find_existing(cell)
    if p is None:
        return None
    with open(p) as f:
        d = json.load(f)
    return {
        "roc_auc": float(d["roc_auc"]),
        "ece": float(d["ece"]),
        "brier_score_loss": float(d["brier_score_loss"]),
        "n_samples": int(d["n_samples"]),
        "results_path": str(p),
    }


# ---------- Per-model run group ----------------------------------------------

def _max_model_len_for(spec: ModelSpec, modes: list[str]) -> int:
    """Big-enough max_model_len to cover the worst mode in this run.

    Reasoning prompts can emit thousands of CoT tokens; everything else is short.
    """
    if any(m in {"reasoning", "reasoning_think"} for m in modes):
        return 8192
    return 2048


def run_model_group(
    spec: ModelSpec,
    modes: list[str],
    backend: str,
    *,
    skip_existing: bool = True,
) -> list[tuple[Cell, dict | None, Optional[Exception]]]:
    """Load `spec` on `backend`, then iterate `modes` reusing the same model."""
    model_path = MODELS_ROOT / spec.name
    print(f"\n=== loading model {model_path} ({backend}) ===", flush=True)

    try:
        if backend == "vllm":
            from folktexts.llm_utils import load_vllm_model
            model, tokenizer = load_vllm_model(
                model_path.as_posix(),
                max_model_len=_max_model_len_for(spec, modes),
                seed=SEED,
            )
        else:
            model, tokenizer = load_model_tokenizer(model_path.as_posix())
    except Exception as exc:
        # Mark every requested mode as failed and move on to the next model.
        import traceback
        tb = traceback.format_exc()
        out_load_fail: list[tuple[Cell, dict | None, Optional[Exception]]] = []
        for mode in modes:
            cell = Cell(model_name=spec.name, mode=mode, backend=backend)
            log_path = LOGS_DIR / f"{cell.slug}.log"
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log_path.write_text(f"Failed to load model {spec.name} ({backend}):\n\n{tb}\n")
            print(f"!! {cell.slug} MODEL-LOAD FAILED ({type(exc).__name__}): {exc}", flush=True)
            out_load_fail.append((cell, None, exc))
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return out_load_fail

    out: list[tuple[Cell, dict | None, Optional[Exception]]] = []
    for mode in modes:
        flags = MODE_FLAGS[mode]
        numeric, use_chat, reasoning, thinking = flags
        cell = Cell(model_name=spec.name, mode=mode, backend=backend)
        existing = _find_existing(cell) if skip_existing else None
        if existing is not None:
            metrics = _load_metrics(cell)
            print(f"-- {cell.slug}: skipping (found existing)", flush=True)
            out.append((cell, metrics, None))
            continue

        print(f"\n>>> {cell.slug}", flush=True)
        try:
            config = BenchmarkConfig(
                numeric_risk_prompting=numeric,
                use_chat_template=use_chat,
                reasoning_prompting=reasoning,
                enable_thinking=thinking,
                batch_size=BATCH_SIZE,
                seed=SEED,
            )
            bench = Benchmark.make_acs_benchmark(
                task_name=TASK,
                model=model,
                tokenizer=tokenizer,
                data_dir=DATA_DIR.as_posix(),
                config=config,
                subsampling=SUBSAMPLING,
                backend=backend,
                model_name_or_path=model_path.as_posix() if backend == "vllm" else None,
            )
            base_results_dir = _result_dir_for(cell).parent
            results_dir = get_or_create_results_dir(
                model_name=spec.name,
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
            print(f"<<< {cell.slug}: AUC={metrics['roc_auc']:.4f} "
                  f"ECE={metrics['ece']:.4f}", flush=True)
            out.append((cell, metrics, None))
            del bench
        except Exception as exc:
            tb = traceback.format_exc()
            log_path = LOGS_DIR / f"{cell.slug}.log"
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log_path.write_text(f"Failed cell: {cell.slug}\n\n{tb}\n")
            print(f"!! {cell.slug} FAILED ({type(exc).__name__}): {exc}", flush=True)
            out.append((cell, None, exc))

    del model, tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return out


# ---------- Report -----------------------------------------------------------

GATE_AUC = 0.015        # Phase 4 acceptance is slightly looser than Phase 1.
GATE_ECE = 0.025


def _delta(a: float | None, b: float | None, gate: float) -> str:
    if a is None or b is None:
        return "—"
    d = a - b
    cell = f"{d:+.3f}"
    return f"**{cell}**" if abs(d) > gate else cell


def _fmt(x: float | None) -> str:
    return f"{x:.3f}" if x is not None else "—"


def write_report(specs: list[ModelSpec], backends: list[str]) -> None:
    """Walk RESULTS_DIR and emit a vllm-vs-transformers comparison table."""
    out_path = RESULTS_DIR / "REPORT.md"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    lines: list[str] = []
    lines.append("# Extended sweep: modern + thinking models — vLLM vs transformers")
    lines.append("")
    lines.append(
        f"Per (model, mode) pair: AUC and ECE on transformers vs vLLM. "
        f"`Δ = vllm − transformers`. Cells with `|ΔAUC|>{GATE_AUC}` or "
        f"`|ΔECE|>{GATE_ECE}` are **bolded**.\n"
        f"- task: `{TASK}`, subsampling: `{SUBSAMPLING}`, seed: {SEED}, batch: {BATCH_SIZE}\n"
    )
    lines.append("")
    have_pairs = total = 0

    metric_specs = [("AUC", "roc_auc", GATE_AUC), ("ECE", "ece", GATE_ECE)]
    for title, key, gate in metric_specs:
        lines.append(f"## {title}")
        lines.append("")
        lines.append("| Model | mode | transformers | vLLM | Δ |")
        lines.append("|---|---|---|---|---|")
        for spec in specs:
            for mode in spec.modes:
                tf = _load_metrics(Cell(spec.name, mode, "transformers"))
                vl = _load_metrics(Cell(spec.name, mode, "vllm"))
                if tf is None and vl is None:
                    continue
                if title == "AUC":
                    total += 1
                    if tf and vl:
                        have_pairs += 1
                tf_v = (tf or {}).get(key)
                vl_v = (vl or {}).get(key)
                lines.append(
                    f"| `{spec.name}` | {mode} | {_fmt(tf_v)} | {_fmt(vl_v)} | {_delta(vl_v, tf_v, gate)} |"
                )
        lines.append("")

    lines.insert(4, f"- Coverage: **{have_pairs}/{total}** matched pairs.\n")

    out_path.write_text("\n".join(lines) + "\n")
    print(f"Wrote -> {out_path}  ({have_pairs}/{total} matched pairs)")


# ---------- CLI ---------------------------------------------------------------

def _select_specs(args) -> list[ModelSpec]:
    base = TIERS[args.tier]
    if args.models:
        wanted = {m.strip() for m in args.models.split(",") if m.strip()}
        base = [s for s in base if any(w in s.name for w in wanted)]
        if not base:
            raise SystemExit(
                f"No models in --tier {args.tier} matched --models {args.models!r}"
            )
    return base


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tier", choices=sorted(TIERS), default="tier1",
                        help="Which model menu to sweep (default tier1).")
    parser.add_argument("--models", default=None,
                        help="Comma-separated substring filter applied to the tier list.")
    parser.add_argument("--modes", default=None,
                        help=f"Subset of {list(MODE_FLAGS)} (default: each model's full mode list).")
    parser.add_argument("--backends", default=None,
                        help=f"Comma-separated subset of {DEFAULT_BACKENDS}.")
    parser.add_argument("--no-skip", action="store_true",
                        help="Re-run cells even if a matching JSON exists.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--report-only", action="store_true",
                        help="Skip running anything; rebuild REPORT.md only.")
    args = parser.parse_args()

    specs = _select_specs(args)
    if args.modes:
        wanted_modes = [m.strip() for m in args.modes.split(",") if m.strip()]
        unknown = [m for m in wanted_modes if m not in MODE_FLAGS]
        if unknown:
            raise SystemExit(f"Unknown mode(s): {unknown}. Known: {list(MODE_FLAGS)}")
        specs = [
            ModelSpec(s.name, tuple(m for m in s.modes if m in wanted_modes), s.notes)
            for s in specs
        ]
        specs = [s for s in specs if s.modes]

    backends = (
        [b.strip() for b in args.backends.split(",") if b.strip()]
        if args.backends else list(DEFAULT_BACKENDS)
    )
    unknown_b = [b for b in backends if b not in DEFAULT_BACKENDS]
    if unknown_b:
        raise SystemExit(f"Unknown backend(s): {unknown_b}. Known: {DEFAULT_BACKENDS}")

    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(levelname)s %(message)s")

    if args.report_only:
        write_report(specs, backends)
        return 0

    print(f"Planned: tier={args.tier}, models={[s.name for s in specs]}, "
          f"modes per model varies, backends={backends}")
    cells: list[Cell] = []
    for backend in backends:
        for spec in specs:
            for mode in spec.modes:
                cells.append(Cell(spec.name, mode, backend))
    print(f"Total cells: {len(cells)}")
    for c in cells:
        marker = "  [SKIP — existing]" if (not args.no_skip and _find_existing(c)) else ""
        print(f"  - {c.slug}{marker}")
    if args.dry_run:
        return 0

    fails: list[str] = []
    for backend in backends:
        for spec in specs:
            results = run_model_group(spec, list(spec.modes), backend, skip_existing=not args.no_skip)
            fails.extend(c.slug for c, m, exc in results if exc is not None)

    write_report(specs, backends)

    if fails:
        print(f"\n{len(fails)} cells failed: {fails}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
