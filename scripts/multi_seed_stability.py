"""Multi-seed stability check for the vLLM migration.

For a small set of representative (model, mode) cells, run the benchmark
across both backends and a handful of seeds. The migration is acceptable
iff the cross-backend AUC delta is no larger than the natural cross-seed
AUC standard deviation on either backend.

Usage:
    python scripts/multi_seed_stability.py            # full sweep
    python scripts/multi_seed_stability.py --report-only
    python scripts/multi_seed_stability.py --models meta-llama--Meta-Llama-3-8B-Instruct
    python scripts/multi_seed_stability.py --backends vllm,transformers
    python scripts/multi_seed_stability.py --seeds 42,1337
"""
from __future__ import annotations

import argparse
import gc
import json
import logging
import statistics
import sys
import traceback
from dataclasses import dataclass, asdict
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
RESULTS_DIR = REPO_ROOT / "results" / "multi-seed-stability"
LOGS_DIR = RESULTS_DIR / "logs"
TASK = "ACSIncome"
SUBSAMPLING = 0.01      # ~3.3k rows on ACSIncome test
BATCH_SIZE = 16
DEFAULT_SEEDS = [42, 1337, 2024, 31415]
DEFAULT_BACKENDS = ["transformers", "vllm"]


# Mode tag -> (numeric_risk_prompting, use_chat_template, reasoning_prompting, enable_thinking)
MODE_FLAGS: dict[str, tuple[bool, bool, bool, bool]] = {
    "mcq":        (False, False, False, False),
    "chat-mcq":   (False, True,  False, False),
    "chat-numeric": (True, True,  False, False),
    "reasoning":  (False, False, True,  False),   # ReasoningQA, enable_thinking auto
    "reasoning-thinking": (False, False, True, True),  # ReasoningQA + chat thinking on
}

# Per-model mode list. Llama-3-8B-Instruct is the bread-and-butter cell;
# Qwen3-4B-Thinking-2507 exercises the reasoning + thinking path.
DEFAULT_MODELS: dict[str, list[str]] = {
    "meta-llama--Meta-Llama-3-8B-Instruct": ["mcq", "chat-mcq", "chat-numeric", "reasoning"],
    "Qwen--Qwen3-4B-Thinking-2507": ["chat-mcq", "chat-numeric", "reasoning-thinking"],
}


@dataclass(frozen=True)
class Cell:
    model_name: str
    mode: str
    seed: int
    backend: str

    @property
    def slug(self) -> str:
        return f"{self.model_name}__{self.mode}__seed{self.seed}__{self.backend}"


def cells_for(
    models: dict[str, list[str]],
    seeds: list[int],
    backends: list[str],
) -> list[Cell]:
    out: list[Cell] = []
    for backend in backends:
        for model_name, modes in models.items():
            for mode in modes:
                for seed in seeds:
                    out.append(Cell(model_name=model_name, mode=mode, seed=seed, backend=backend))
    return out


# ---------- Idempotency: skip cells whose JSON we've already computed ---------

def _result_dir_for(cell: Cell) -> Path:
    return RESULTS_DIR / cell.backend / f"model-{cell.model_name}_task-{TASK}"


def _find_existing(cell: Cell) -> Optional[Path]:
    parent = _result_dir_for(cell)
    if not parent.exists():
        return None
    flags = MODE_FLAGS[cell.mode]
    numeric, use_chat, reasoning, thinking = flags
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
            and cfg.get("seed") == cell.seed
            and cfg.get("batch_size") == BATCH_SIZE
            and abs((cfg.get("dataset_subsampling") or 0) - SUBSAMPLING) < 1e-6
        ):
            return path
    return None


# ---------- Per-cell execution ------------------------------------------------

def run_cell_group(
    *,
    model_name: str,
    modes: list[str],
    seeds: list[int],
    backend: str,
    skip_existing: bool = True,
) -> list[tuple[Cell, dict | None, Optional[Exception]]]:
    """Load `model_name` on `backend`, then iterate (mode × seed) without
    re-loading. Returns a list of (cell, metrics, exception) triples."""
    model_path = MODELS_ROOT / model_name
    print(f"\n=== loading model {model_path} ({backend}) ===", flush=True)

    if backend == "vllm":
        from folktexts.llm_utils import load_vllm_model
        # Reasoning needs ample output budget; 8k covers Qwen3-Thinking outputs and
        # leaves room for the input prompt. ACSIncome prompts are well under 1k.
        max_model_len = 8192
        model, tokenizer = load_vllm_model(
            model_path.as_posix(),
            max_model_len=max_model_len,
            seed=seeds[0],  # vLLM seed is irrelevant for greedy; use first seed.
        )
    else:
        model, tokenizer = load_model_tokenizer(model_path.as_posix())

    out: list[tuple[Cell, dict | None, Optional[Exception]]] = []
    for mode in modes:
        flags = MODE_FLAGS[mode]
        numeric, use_chat, reasoning, thinking = flags
        for seed in seeds:
            cell = Cell(model_name=model_name, mode=mode, seed=seed, backend=backend)
            existing = _find_existing(cell) if skip_existing else None
            if existing is not None:
                with open(existing) as f:
                    d = json.load(f)
                metrics = {
                    "roc_auc": float(d["roc_auc"]),
                    "ece": float(d["ece"]),
                    "brier_score_loss": float(d["brier_score_loss"]),
                    "n_samples": int(d["n_samples"]),
                    "results_path": str(existing),
                }
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
                    seed=seed,
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
                    model_name=model_name,
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
                print(
                    f"<<< {cell.slug}: AUC={metrics['roc_auc']:.4f} "
                    f"ECE={metrics['ece']:.4f}", flush=True,
                )
                out.append((cell, metrics, None))
                del bench
            except Exception as exc:
                tb = traceback.format_exc()
                log_path = LOGS_DIR / f"{cell.slug}.log"
                log_path.parent.mkdir(parents=True, exist_ok=True)
                log_path.write_text(f"Failed cell: {cell.slug}\n\n{tb}\n")
                print(f"!! {cell.slug} FAILED ({type(exc).__name__}): {exc}", flush=True)
                out.append((cell, None, exc))

    # Free GPU memory before next model load.
    del model, tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return out


# ---------- Aggregation + report ----------------------------------------------

def _collect_existing_metrics(
    models: dict[str, list[str]], seeds: list[int], backends: list[str]
) -> dict[Cell, dict | None]:
    out: dict[Cell, dict | None] = {}
    for cell in cells_for(models, seeds, backends):
        path = _find_existing(cell)
        if path is None:
            out[cell] = None
            continue
        with open(path) as f:
            d = json.load(f)
        out[cell] = {
            "roc_auc": float(d["roc_auc"]),
            "ece": float(d["ece"]),
            "brier_score_loss": float(d["brier_score_loss"]),
            "n_samples": int(d["n_samples"]),
            "results_path": str(path),
        }
    return out


def _stats(values: list[float]) -> tuple[float, float]:
    if not values:
        return float("nan"), float("nan")
    if len(values) == 1:
        return values[0], 0.0
    return statistics.mean(values), statistics.stdev(values)


def write_report(
    metrics: dict[Cell, dict | None],
    models: dict[str, list[str]],
    seeds: list[int],
    backends: list[str],
) -> None:
    out_path = RESULTS_DIR / "REPORT.md"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    lines: list[str] = []
    lines.append("# Multi-seed stability: vLLM vs transformers")
    lines.append("")
    lines.append(
        f"Per (model, mode, backend), runs the benchmark across {len(seeds)} seeds "
        f"(`{seeds}`) at `subsampling={SUBSAMPLING}` (~{int(SUBSAMPLING*332901)} rows). "
        "Reports mean ± std for AUC and ECE, and the cross-backend delta of means. "
        "**Acceptance:** `|Δmean(AUC)| ≤ 2 × max(std(AUC) over both backends)`."
    )
    lines.append("")

    # Group rows by (model, mode), columns by backend.
    for model_name, modes in models.items():
        lines.append(f"## `{model_name}`")
        lines.append("")
        lines.append(
            "| Mode | backend | n seeds | AUC mean | AUC std | ECE mean | ECE std |"
        )
        lines.append("|---|---|---|---|---|---|---|")
        for mode in modes:
            for backend in backends:
                values_auc, values_ece = [], []
                for seed in seeds:
                    c = Cell(model_name=model_name, mode=mode, seed=seed, backend=backend)
                    m = metrics.get(c)
                    if m is None:
                        continue
                    values_auc.append(m["roc_auc"])
                    values_ece.append(m["ece"])
                auc_mu, auc_sd = _stats(values_auc)
                ece_mu, ece_sd = _stats(values_ece)
                lines.append(
                    f"| {mode} | {backend} | {len(values_auc)} "
                    f"| {auc_mu:.4f} | {auc_sd:.4f} "
                    f"| {ece_mu:.4f} | {ece_sd:.4f} |"
                )
        lines.append("")

        # Cross-backend acceptance check: |Δmean(AUC)| vs 2× max(std).
        if len(backends) >= 2:
            lines.append("**Cross-backend delta vs cross-seed std (acceptance check)**")
            lines.append("")
            lines.append(
                "| Mode | Δmean(AUC) | 2× max std | gate | Δmean(ECE) |"
            )
            lines.append("|---|---|---|---|---|")
            for mode in modes:
                # First two backends only — we only have transformers vs vllm here.
                b0, b1 = backends[0], backends[1]
                aucs = {b: [] for b in (b0, b1)}
                eces = {b: [] for b in (b0, b1)}
                for seed in seeds:
                    for b in (b0, b1):
                        m = metrics.get(Cell(model_name, mode, seed, b))
                        if m is None:
                            continue
                        aucs[b].append(m["roc_auc"])
                        eces[b].append(m["ece"])
                if not aucs[b0] or not aucs[b1]:
                    continue
                _, sd0 = _stats(aucs[b0])
                _, sd1 = _stats(aucs[b1])
                delta_auc = statistics.mean(aucs[b1]) - statistics.mean(aucs[b0])
                bound = 2.0 * max(sd0, sd1)
                gate = "✅" if abs(delta_auc) <= bound else "❌"
                delta_ece = statistics.mean(eces[b1]) - statistics.mean(eces[b0])
                lines.append(
                    f"| {mode} | {delta_auc:+.4f} | {bound:.4f} | {gate} | {delta_ece:+.4f} |"
                )
            lines.append("")

    out_path.write_text("\n".join(lines) + "\n")
    print(f"\nWrote report -> {out_path}")


# ---------- CLI ---------------------------------------------------------------

def _parse_models(arg: str | None) -> dict[str, list[str]]:
    if not arg:
        return DEFAULT_MODELS
    chosen = [m.strip() for m in arg.split(",") if m.strip()]
    out: dict[str, list[str]] = {}
    for m in chosen:
        if m not in DEFAULT_MODELS:
            raise SystemExit(
                f"Unknown model '{m}'. Add it to DEFAULT_MODELS first. "
                f"Known: {list(DEFAULT_MODELS)}"
            )
        out[m] = DEFAULT_MODELS[m]
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models", default=None, help="Comma-separated subset.")
    parser.add_argument("--backends", default=None,
                        help=f"Comma-separated subset of {DEFAULT_BACKENDS}.")
    parser.add_argument("--seeds", default=None,
                        help=f"Comma-separated seeds (default {DEFAULT_SEEDS}).")
    parser.add_argument("--modes", default=None,
                        help=f"Comma-separated subset of {list(MODE_FLAGS)}; "
                             "applied to each model's default mode list.")
    parser.add_argument("--no-skip", action="store_true",
                        help="Re-run cells even if a matching JSON exists.")
    parser.add_argument("--dry-run", action="store_true",
                        help="List cells then exit.")
    parser.add_argument("--report-only", action="store_true",
                        help="Skip running anything; rebuild REPORT.md only.")
    args = parser.parse_args()

    models = _parse_models(args.models)
    if args.modes:
        wanted = [m.strip() for m in args.modes.split(",") if m.strip()]
        unknown = [m for m in wanted if m not in MODE_FLAGS]
        if unknown:
            raise SystemExit(f"Unknown mode(s): {unknown}. Known: {list(MODE_FLAGS)}")
        models = {
            name: [m for m in defmodes if m in wanted]
            for name, defmodes in models.items()
        }
        models = {n: ms for n, ms in models.items() if ms}
    backends = (
        [b.strip() for b in args.backends.split(",") if b.strip()]
        if args.backends else list(DEFAULT_BACKENDS)
    )
    seeds = (
        [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
        if args.seeds else list(DEFAULT_SEEDS)
    )
    unknown_b = [b for b in backends if b not in DEFAULT_BACKENDS]
    if unknown_b:
        raise SystemExit(f"Unknown backend(s): {unknown_b}. Known: {DEFAULT_BACKENDS}")

    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(levelname)s %(message)s")

    if args.report_only:
        metrics = _collect_existing_metrics(models, seeds, backends)
        write_report(metrics, models, seeds, backends)
        return 0

    cells = cells_for(models, seeds, backends)
    print(f"Planned cells ({len(cells)}):")
    for c in cells:
        marker = "  [SKIP — existing]" if (not args.no_skip and _find_existing(c)) else ""
        print(f"  - {c.slug}{marker}")
    if args.dry_run:
        return 0

    # Group by (backend, model) to amortise the model-load cost.
    fails: list[str] = []
    for backend in backends:
        for model_name, modes in models.items():
            results = run_cell_group(
                model_name=model_name,
                modes=modes,
                seeds=seeds,
                backend=backend,
                skip_existing=not args.no_skip,
            )
            fails.extend(c.slug for c, m, exc in results if exc is not None)

    metrics = _collect_existing_metrics(models, seeds, backends)
    write_report(metrics, models, seeds, backends)

    if fails:
        print(f"\n{len(fails)} cells failed: {fails}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
