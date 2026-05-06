"""Reproduce Table 1 of arxiv 2407.14614v3 on the current code-base.

For each (model, prompting_strategy) combination:
    - load model + tokenizer
    - build a Benchmark with the chosen config (numeric / MCQ, chat or zero-shot)
    - run on a 10k subsample of the ACSIncome test set
    - record AUC / Accuracy / ECE in-memory and to disk

After the sweep, write `results/paper-reproduction/TABLE1_REPRODUCTION.md`
which compares our numbers to the paper's verbatim Table 1 values.

Usage:
    python scripts/reproduce_table1.py                     # full sweep, smallest first
    python scripts/reproduce_table1.py --models name1,name2  # subset
    python scripts/reproduce_table1.py --report-only       # rebuild markdown only
    python scripts/reproduce_table1.py --dry-run           # list combos and exit
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
DEFAULT_RESULTS_SUBDIR = "paper-reproduction"
# Mutated by main() based on --results-subdir / --backend so module-level
# helpers (write_report, _find_existing_result) write/read consistently.
RESULTS_DIR = REPO_ROOT / "results" / DEFAULT_RESULTS_SUBDIR
LOGS_DIR = RESULTS_DIR / "logs"
TASK = "ACSIncome"
SUBSAMPLING = 0.03            # ~10k rows on ACSIncome test
SEED = 42
BATCH_SIZE = 16

# Verbatim from arxiv 2407.14614v3 Table 1 (ACSIncome, 160k held-out test, M=10).
# Tuple order: (AUC, Accuracy, ECE).
PAPER_TABLE1: dict[str, dict[str, tuple[float, float, float]]] = {
    "google--gemma-1.1-2b-it":              {"mcq": (0.73, 0.37, 0.63), "num": (0.50, 0.37, 0.28)},
    "mistralai--Mistral-7B-v0.1":           {"mcq": (0.80, 0.73, 0.20), "num": (0.75, 0.49, 0.36)},
    "mistralai--Mistral-7B-Instruct-v0.2":  {"mcq": (0.83, 0.77, 0.21), "num": (0.83, 0.70, 0.16)},
    "meta-llama--Meta-Llama-3-8B":          {"mcq": (0.81, 0.38, 0.25), "num": (0.63, 0.40, 0.14)},
    "meta-llama--Meta-Llama-3-8B-Instruct": {"mcq": (0.85, 0.62, 0.32), "num": (0.81, 0.67, 0.23)},
    "01-ai--Yi-34B":                        {"mcq": (0.85, 0.62, 0.25), "num": (0.83, 0.61, 0.15)},
    "01-ai--Yi-34B-Chat":                   {"mcq": (0.86, 0.72, 0.19), "num": (0.80, 0.48, 0.22)},
    "meta-llama--Meta-Llama-3-70B-Instruct": {"mcq": (0.86, 0.69, 0.27), "num": (0.84, 0.67, 0.25)},
}

# Models that ALSO get the new chat-template column.
CHAT_TEMPLATE_MODELS: set[str] = {
    "meta-llama--Meta-Llama-3-8B-Instruct",
    "meta-llama--Meta-Llama-3-70B-Instruct",
}

# Sweep order: smallest first so failures don't block earlier work.
MODEL_ORDER: list[str] = [
    "google--gemma-1.1-2b-it",
    "mistralai--Mistral-7B-v0.1",
    "mistralai--Mistral-7B-Instruct-v0.2",
    "meta-llama--Meta-Llama-3-8B",
    "meta-llama--Meta-Llama-3-8B-Instruct",
    "01-ai--Yi-34B",
    "01-ai--Yi-34B-Chat",
    "meta-llama--Meta-Llama-3-70B-Instruct",
]


# ---------- Run record --------------------------------------------------------

@dataclass(frozen=True)
class Combo:
    model_name: str
    numeric: bool
    use_chat: bool

    @property
    def slug(self) -> str:
        prompt = "numeric" if self.numeric else "mcq"
        chat = "chat" if self.use_chat else "noChat"
        return f"{self.model_name}__{prompt}__{chat}"


def all_combos(model_names: Iterable[str]) -> list[Combo]:
    out: list[Combo] = []
    for name in model_names:
        for numeric in (False, True):  # MCQ first, then numeric
            out.append(Combo(name, numeric, use_chat=False))
            if name in CHAT_TEMPLATE_MODELS:
                out.append(Combo(name, numeric, use_chat=True))
    return out


# ---------- Idempotency: find a previously-run JSON for this combo -------------

def _find_existing_result(combo: Combo) -> Optional[Path]:
    """Look for any results.bench-*.json under RESULTS_DIR whose config matches `combo`.

    We can't predict the hash without instantiating the Benchmark (it depends on
    dataset hash, model hash, config hash), so we walk and inspect each JSON.
    """
    parent = RESULTS_DIR / f"model-{combo.model_name}_task-{TASK}"
    if not parent.exists():
        return None
    for path in parent.glob("**/results.bench-*.json"):
        try:
            with open(path) as f:
                data = json.load(f)
        except Exception:
            continue
        cfg = data.get("config", {})
        if (
            cfg.get("numeric_risk_prompting") == combo.numeric
            and cfg.get("use_chat_template") == combo.use_chat
            and cfg.get("seed") == SEED
            and cfg.get("batch_size") == BATCH_SIZE
            and abs((cfg.get("dataset_subsampling") or 0) - SUBSAMPLING) < 1e-6
        ):
            return path
    return None


# ---------- Per-combo execution ----------------------------------------------

def run_combo(combo: Combo, *, backend: str = "transformers") -> dict:
    """Execute one (model, numeric, use_chat) combo. Returns the metrics dict.

    `backend` selects between the HuggingFace transformers path (default for
    backwards compatibility with existing recorded numbers) and the vLLM path.
    """
    model_path = MODELS_ROOT / combo.model_name
    print(f"\n=== {combo.slug} ===  backend={backend}  loading {model_path}", flush=True)

    if backend == "vllm":
        from folktexts.llm_utils import load_vllm_model
        # Reasoning prompts need a much bigger output budget; reproduce_table1
        # runs only MCQ/numeric, so a modest max_model_len is plenty.
        max_model_len = 2048
        model, tokenizer = load_vllm_model(
            model_path.as_posix(),
            max_model_len=max_model_len,
            seed=SEED,
        )
    else:
        model, tokenizer = load_model_tokenizer(model_path.as_posix())

    config = BenchmarkConfig(
        numeric_risk_prompting=combo.numeric,
        use_chat_template=combo.use_chat,
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
    out_dir = get_or_create_results_dir(
        model_name=combo.model_name,
        task_name=TASK,
        results_root_dir=RESULTS_DIR.as_posix(),
    )
    bench.run(results_root_dir=out_dir, fit_threshold=False)  # tau=0.5 -- paper convention
    res = bench.results
    metrics = {
        "roc_auc": float(res["roc_auc"]),
        "accuracy": float(res["accuracy"]),
        "ece": float(res["ece"]),
        "brier_score_loss": float(res["brier_score_loss"]),
        "n_samples": int(res["n_samples"]),
        "results_path": str(Path(res["results_dir"]) / f"results.bench-{res['benchmark_hash']}.json"),
    }
    print(f"=== {combo.slug} ===  AUC={metrics['roc_auc']:.4f}  "
          f"Acc={metrics['accuracy']:.4f}  ECE={metrics['ece']:.4f}", flush=True)

    del bench, model, tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return metrics


# ---------- Markdown report ---------------------------------------------------

def _load_metrics_for_combo(combo: Combo) -> Optional[dict]:
    p = _find_existing_result(combo)
    if p is None:
        return None
    with open(p) as f:
        d = json.load(f)
    return {
        "roc_auc": d.get("roc_auc"),
        "accuracy": d.get("accuracy"),
        "ece": d.get("ece"),
        "n_samples": d.get("n_samples"),
        "results_path": str(p),
    }


def _fmt(x: Optional[float]) -> str:
    if x is None:
        return "—"
    return f"{x:.3f}"


def _fmt_delta(ours: Optional[float], paper: Optional[float]) -> str:
    if ours is None or paper is None:
        return "—"
    d = ours - paper
    cell = f"{d:+.3f}"
    if abs(d) > 0.02:
        cell = f"**{cell}**"
    return cell


def write_report() -> None:
    """Walk RESULTS_DIR, gather all (model, numeric, use_chat) cells, and write markdown."""
    out_path = RESULTS_DIR / "TABLE1_REPRODUCTION.md"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows: dict[tuple[str, bool, bool], dict] = {}
    for combo in all_combos(MODEL_ORDER):
        cell = _load_metrics_for_combo(combo)
        if cell is not None:
            rows[(combo.model_name, combo.numeric, combo.use_chat)] = cell

    metric_specs = [
        ("AUC", "roc_auc", 0),
        ("Accuracy", "accuracy", 1),
        ("ECE", "ece", 2),
    ]

    lines: list[str] = []
    lines.append("# Table 1 reproduction: arxiv 2407.14614v3 on the current code-base")
    lines.append("")
    lines.append(
        "Generated by `scripts/reproduce_table1.py`. Each row is a model from Table 1; "
        "each metric (AUC / Accuracy / ECE) gets its own table with paper-vs-ours-vs-Δ "
        "columns and a new chat-template column for the instruct models that received it."
    )
    lines.append("")
    lines.append(
        f"- ACS task: `{TASK}` | held-out test subsample: **n≈{int(SUBSAMPLING*332901)}** "
        f"(`subsampling={SUBSAMPLING}`) vs paper's n=160,000\n"
        f"- Random seed: {SEED} | batch size: {BATCH_SIZE} | classification threshold: 0.5\n"
        f"- Cells with `|Δ| > 0.02` are **bolded** as candidate regressions.\n"
        f"- Yi-34B-Chat is mapped to the paper's *Yi 34B (it)* row "
        f"(only Yi 34B chat checkpoint that existed at submission time).\n"
        f"- Branch: `bugfix/vocab-and-prefill` (commits `bede509`, `27320fe`, `605a0b8`)."
    )
    lines.append("")

    for title, key, idx in metric_specs:
        lines.append(f"## {title}")
        lines.append("")
        header = ("| Model | paper-MCQ | ours-MCQ | Δ | paper-Numeric | ours-Numeric | Δ "
                  "| ours-chat-MCQ | ours-chat-Numeric |")
        sep = ("|---|---|---|---|---|---|---|---|---|")
        lines.append(header)
        lines.append(sep)
        for name in MODEL_ORDER:
            paper = PAPER_TABLE1[name]
            paper_mcq = paper["mcq"][idx]
            paper_num = paper["num"][idx]
            ours_mcq = rows.get((name, False, False), {}).get(key)
            ours_num = rows.get((name, True, False), {}).get(key)
            chat_mcq = rows.get((name, False, True), {}).get(key) if name in CHAT_TEMPLATE_MODELS else None
            chat_num = rows.get((name, True, True), {}).get(key) if name in CHAT_TEMPLATE_MODELS else None
            lines.append(
                f"| `{name}` "
                f"| {paper_mcq:.2f} | {_fmt(ours_mcq)} | {_fmt_delta(ours_mcq, paper_mcq)} "
                f"| {paper_num:.2f} | {_fmt(ours_num)} | {_fmt_delta(ours_num, paper_num)} "
                f"| {_fmt(chat_mcq) if name in CHAT_TEMPLATE_MODELS else 'n/a'} "
                f"| {_fmt(chat_num) if name in CHAT_TEMPLATE_MODELS else 'n/a'} |"
            )
        lines.append("")

    # Footnote: how to read deltas given subsample SE.
    lines.append("---")
    lines.append("")
    lines.append(
        "**Sampling-noise footnote.** With n≈10k held-out and AUC≈0.8, the standard "
        "error of AUC is roughly 0.005, so |Δ| up to ~0.01 is sampling noise rather "
        "than a real regression. Re-run the combo with a larger `--subsampling` to "
        "tighten the comparison."
    )
    lines.append("")

    out_path.write_text("\n".join(lines) + "\n")
    print(f"\nWrote report -> {out_path}")


# ---------- CLI ---------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models", default=None,
                        help="Comma-separated subset of model dirnames to run. Defaults to all.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print the list of combos that would run, then exit.")
    parser.add_argument("--report-only", action="store_true",
                        help="Skip running anything; just rebuild TABLE1_REPRODUCTION.md from existing JSONs.")
    parser.add_argument("--no-skip", action="store_true",
                        help="Re-run combos even if a matching results JSON already exists.")
    parser.add_argument("--backend", choices=["transformers", "vllm"], default="transformers",
                        help="Local inference backend (default: transformers, to match recorded numbers).")
    parser.add_argument("--results-subdir", default=None,
                        help=("Subdirectory under results/ to read/write. Defaults to "
                              "'paper-reproduction' for transformers and "
                              "'paper-reproduction-vllm' for vllm so a vLLM sweep doesn't "
                              "overwrite the recorded transformers numbers."))
    args = parser.parse_args()

    # Pick the results directory based on backend (overridable via --results-subdir).
    global RESULTS_DIR, LOGS_DIR
    subdir = args.results_subdir or (
        "paper-reproduction-vllm" if args.backend == "vllm" else DEFAULT_RESULTS_SUBDIR
    )
    RESULTS_DIR = REPO_ROOT / "results" / subdir
    LOGS_DIR = RESULTS_DIR / "logs"

    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.WARNING,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    if args.report_only:
        write_report()
        return 0

    selected_names = (
        [m.strip() for m in args.models.split(",") if m.strip()]
        if args.models else list(MODEL_ORDER)
    )
    unknown = [m for m in selected_names if m not in PAPER_TABLE1]
    if unknown:
        print(f"!! Unknown model(s): {unknown}. Known: {list(PAPER_TABLE1)}", file=sys.stderr)
        return 2

    combos = all_combos(selected_names)
    print(f"Planned combos ({len(combos)}):")
    for c in combos:
        existing = _find_existing_result(c)
        marker = "  [SKIP — found existing]" if (existing and not args.no_skip) else ""
        print(f"  - {c.slug}{marker}")
    if args.dry_run:
        return 0

    fails: list[str] = []
    for c in combos:
        if not args.no_skip and _find_existing_result(c) is not None:
            print(f"-- {c.slug}: skipping (found existing results JSON)")
            continue
        try:
            run_combo(c, backend=args.backend)
        except Exception as exc:  # noqa: BLE001
            tb = traceback.format_exc()
            log_path = LOGS_DIR / f"{c.slug}.log"
            log_path.write_text(f"Failed combo: {c.slug}\n\n{tb}\n")
            print(f"!! {c.slug} FAILED ({type(exc).__name__}: {exc}); log at {log_path}", flush=True)
            fails.append(c.slug)
            # Free GPU memory after a failure too.
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    write_report()

    if fails:
        print(f"\n{len(fails)} combos failed: {fails}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
