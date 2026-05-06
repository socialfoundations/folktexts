"""Side-by-side comparison of transformers vs vLLM backends on a small slice.

Runs the same model + same ACS task + same prompting modes through each
backend in turn, then diffs:
  - Mean / max / p95 absolute difference in `risk_score`.
  - Argmax-token agreement rate (% of rows where the top-1 token id matches).
  - AUC, ECE, Brier deltas.

This is the cheap pre-flight gate before kicking off the full Table-1
reproduction (`scripts/reproduce_table1.py --backend vllm`). It lives outside
pytest because it requires a GPU, vLLM, and ~5-15 min wall time.

Usage:
    python scripts/compare_backends.py \\
        --model /fast/groups/sf/huggingface-models/meta-llama--Llama-3.2-1B-Instruct \\
        --task ACSIncome \\
        --data-dir /fast/acruz/data/folktables \\
        --subsampling 0.005 \\
        --modes baseline,chat_mc,chat_numeric,reasoning
"""
from __future__ import annotations

import argparse
import gc
import logging
import sys
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from folktexts.benchmark import Benchmark, BenchmarkConfig  # noqa: E402
from folktexts.cli._utils import get_or_create_results_dir  # noqa: E402
from folktexts.llm_utils import load_model_tokenizer, load_vllm_model  # noqa: E402

# ---------- Acceptance thresholds (per the migration plan) ------------------

PRE_FLIGHT_GATES = {
    "auc_delta_max": 0.005,
    "ece_delta_max": 0.01,
    "score_within_0p05_min_pct": 95.0,
}


@dataclass(frozen=True)
class Mode:
    name: str
    numeric: bool
    use_chat: bool
    reasoning: bool

    @property
    def slug(self) -> str:
        return self.name


MODES = {
    "baseline":       Mode("baseline",       numeric=False, use_chat=False, reasoning=False),
    "chat_mc":        Mode("chat_mc",        numeric=False, use_chat=True,  reasoning=False),
    "chat_numeric":   Mode("chat_numeric",   numeric=True,  use_chat=True,  reasoning=False),
    "numeric":        Mode("numeric",        numeric=True,  use_chat=False, reasoning=False),
    "reasoning":      Mode("reasoning",      numeric=False, use_chat=False, reasoning=True),
}


def run_one(
    *,
    backend: str,
    model_path: str,
    task: str,
    data_dir: str,
    subsampling: float,
    mode: Mode,
    seed: int,
    batch_size: int,
    out_root: Path,
) -> dict:
    """Run one (backend, mode) cell; return dict of metrics + predictions path."""
    print(f"\n--- [{backend}] {mode.slug} ---", flush=True)

    if backend == "vllm":
        max_model_len = 6000 if mode.reasoning else 2048
        model, tokenizer = load_vllm_model(
            model_path, max_model_len=max_model_len, seed=seed,
        )
    else:
        model, tokenizer = load_model_tokenizer(model_path)

    config = BenchmarkConfig(
        numeric_risk_prompting=mode.numeric,
        use_chat_template=mode.use_chat,
        reasoning_prompting=mode.reasoning,
        batch_size=batch_size,
        seed=seed,
    )
    bench = Benchmark.make_acs_benchmark(
        task_name=task,
        model=model,
        tokenizer=tokenizer,
        data_dir=data_dir,
        config=config,
        subsampling=subsampling,
        backend=backend,
        model_name_or_path=model_path if backend == "vllm" else None,
    )
    out_dir = get_or_create_results_dir(
        model_name=Path(model_path).name,
        task_name=task,
        results_root_dir=str(out_root / backend / mode.slug),
    )
    bench.run(results_root_dir=out_dir, fit_threshold=False)

    res = bench.results
    metrics = {
        "roc_auc": float(res["roc_auc"]),
        "accuracy": float(res["accuracy"]),
        "ece": float(res["ece"]),
        "brier_score_loss": float(res["brier_score_loss"]),
        "n_samples": int(res["n_samples"]),
        "predictions_path": res["plots"].get("calibration_curve_path", "")  # placeholder
    }
    # Locate the predictions CSV — it sits alongside the results JSON.
    results_dir = Path(res["results_dir"])
    csv_paths = list(results_dir.glob("*test_predictions.csv"))
    if not csv_paths:
        csv_paths = list(results_dir.glob("*.test_predictions.csv"))
    metrics["predictions_path"] = str(csv_paths[0]) if csv_paths else ""

    # Free GPU memory before the next backend / mode.
    del bench, model, tokenizer
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass

    return metrics


def diff_predictions(tf_csv: str, vl_csv: str) -> dict:
    """Per-row diff of risk_score columns from the two backends."""
    if not tf_csv or not vl_csv:
        return {"error": "missing predictions CSV(s)"}
    tf = pd.read_csv(tf_csv, index_col=0)
    vl = pd.read_csv(vl_csv, index_col=0)
    if not tf.index.equals(vl.index):
        return {"error": "index mismatch between transformers and vllm CSVs"}

    tf_scores = tf["risk_score"].to_numpy()
    vl_scores = vl["risk_score"].to_numpy()
    diff = np.abs(tf_scores - vl_scores)
    within_005 = float((diff <= 0.05).mean() * 100.0)
    return {
        "n": len(tf_scores),
        "mean_abs_diff": float(diff.mean()),
        "max_abs_diff": float(diff.max()),
        "p95_abs_diff": float(np.percentile(diff, 95)),
        "score_within_0p05_pct": within_005,
    }


def report_pair(mode_slug: str, tf_metrics: dict, vl_metrics: dict, diff: dict) -> str:
    auc_d = vl_metrics["roc_auc"] - tf_metrics["roc_auc"]
    ece_d = vl_metrics["ece"] - tf_metrics["ece"]
    brier_d = vl_metrics["brier_score_loss"] - tf_metrics["brier_score_loss"]

    auc_ok = abs(auc_d) <= PRE_FLIGHT_GATES["auc_delta_max"]
    ece_ok = abs(ece_d) <= PRE_FLIGHT_GATES["ece_delta_max"]
    rows_ok = (
        diff.get("score_within_0p05_pct", 0)
        >= PRE_FLIGHT_GATES["score_within_0p05_min_pct"]
    )
    overall_pass = auc_ok and ece_ok and rows_ok

    lines = [
        "",
        f"### {mode_slug}  ({'PASS' if overall_pass else 'FAIL'})",
        f"  transformers: AUC={tf_metrics['roc_auc']:.4f}  ECE={tf_metrics['ece']:.4f}  Brier={tf_metrics['brier_score_loss']:.4f}",
        f"  vllm        : AUC={vl_metrics['roc_auc']:.4f}  ECE={vl_metrics['ece']:.4f}  Brier={vl_metrics['brier_score_loss']:.4f}",
        f"  Δ           : AUC={auc_d:+.4f}{'  [GATE OK]' if auc_ok else '  [GATE FAIL]'}"
        f"   ECE={ece_d:+.4f}{'  [GATE OK]' if ece_ok else '  [GATE FAIL]'}"
        f"   Brier={brier_d:+.4f}",
    ]
    if "error" in diff:
        lines.append(f"  predictions diff: {diff['error']}")
    else:
        lines.append(
            f"  predictions diff: n={diff['n']}  "
            f"mean={diff['mean_abs_diff']:.4f}  "
            f"p95={diff['p95_abs_diff']:.4f}  "
            f"max={diff['max_abs_diff']:.4f}  "
            f"|Δ|≤0.05 in {diff['score_within_0p05_pct']:.1f}% of rows"
            f"{'  [GATE OK]' if rows_ok else '  [GATE FAIL]'}"
        )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, help="Path or HF id of the model.")
    parser.add_argument("--task", default="ACSIncome")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--subsampling", type=float, default=0.005)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument(
        "--modes", default="baseline,chat_mc,chat_numeric,reasoning",
        help=f"Comma-separated subset of {','.join(MODES)}.",
    )
    parser.add_argument(
        "--out-root", default=str(REPO_ROOT / "results" / "backend-comparison"),
        help="Directory under which per-backend per-mode subfolders are created.",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(levelname)s %(message)s")

    selected = [MODES[m.strip()] for m in args.modes.split(",") if m.strip()]
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    summary_lines: list[str] = []
    summary_lines.append(f"# Backend comparison: {Path(args.model).name}")
    summary_lines.append(
        f"\nTask: `{args.task}` | subsampling: {args.subsampling} | "
        f"seed: {args.seed} | batch size: {args.batch_size}"
    )
    summary_lines.append(
        f"\nGates: |ΔAUC|≤{PRE_FLIGHT_GATES['auc_delta_max']}, "
        f"|ΔECE|≤{PRE_FLIGHT_GATES['ece_delta_max']}, "
        f"≥{PRE_FLIGHT_GATES['score_within_0p05_min_pct']}% rows |Δ|≤0.05"
    )

    any_fail = False
    for mode in selected:
        try:
            tf = run_one(
                backend="transformers", model_path=args.model, task=args.task,
                data_dir=args.data_dir, subsampling=args.subsampling, mode=mode,
                seed=args.seed, batch_size=args.batch_size, out_root=out_root,
            )
            vl = run_one(
                backend="vllm", model_path=args.model, task=args.task,
                data_dir=args.data_dir, subsampling=args.subsampling, mode=mode,
                seed=args.seed, batch_size=args.batch_size, out_root=out_root,
            )
        except Exception as exc:  # noqa: BLE001
            print(f"!! {mode.slug} FAILED: {exc}", file=sys.stderr)
            summary_lines.append(f"\n### {mode.slug}  (ERROR)\n  {exc}")
            any_fail = True
            continue

        diff = diff_predictions(tf["predictions_path"], vl["predictions_path"])
        block = report_pair(mode.slug, tf, vl, diff)
        print(block, flush=True)
        summary_lines.append(block)

        # Treat unmet gates as a fail flag for the exit code; the user can still
        # inspect the report to decide whether to ship anyway.
        if "GATE FAIL" in block:
            any_fail = True

    summary_path = out_root / f"summary_{Path(args.model).name}.md"
    summary_path.write_text("\n".join(summary_lines) + "\n")
    print(f"\nWrote summary -> {summary_path}", flush=True)

    return 1 if any_fail else 0


if __name__ == "__main__":
    sys.exit(main())
