"""Cross-backend Table 1 diff: vLLM vs transformers.

After running `scripts/reproduce_table1.py` once per backend (writing to
`results/paper-reproduction/` and `results/paper-reproduction-vllm/`), this
script emits a single markdown report comparing AUC / ECE / Brier per cell
side by side, flagging any cell that exceeds the migration acceptance gates.

Usage:
    python scripts/compare_table1_backends.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

# Reuse the model list / paper baseline from reproduce_table1 so the two
# reports stay in sync without us copy-pasting model lists.
from scripts.reproduce_table1 import (  # noqa: E402
    CHAT_TEMPLATE_MODELS,
    MODEL_ORDER,
    PAPER_TABLE1,
    SEED,
    SUBSAMPLING,
    BATCH_SIZE,
    TASK,
    Combo,
    all_combos,
)

# Acceptance gates from the migration plan (Phase F, Table 1 stage).
GATE_AUC = 0.01
GATE_ECE = 0.02

TF_DIR = REPO_ROOT / "results" / "paper-reproduction"
VL_DIR = REPO_ROOT / "results" / "paper-reproduction-vllm"
OUT_PATH = VL_DIR / "TABLE1_BACKEND_COMPARISON.md"


def _find_result_json(root: Path, combo: Combo) -> Path | None:
    parent = root / f"model-{combo.model_name}_task-{TASK}"
    if not parent.exists():
        return None
    for path in parent.glob("**/results.bench-*.json"):
        try:
            with open(path) as f:
                cfg = json.load(f).get("config", {})
        except Exception:
            continue
        if (
            cfg.get("numeric_risk_prompting") == combo.numeric
            and cfg.get("use_chat_template") == combo.use_chat
            and cfg.get("seed") == SEED
            and cfg.get("batch_size") == BATCH_SIZE
            and abs((cfg.get("dataset_subsampling") or 0) - SUBSAMPLING) < 1e-6
        ):
            return path
    return None


def _load_metrics(root: Path, combo: Combo) -> dict | None:
    p = _find_result_json(root, combo)
    if p is None:
        return None
    with open(p) as f:
        d = json.load(f)
    return {
        "roc_auc": d.get("roc_auc"),
        "ece": d.get("ece"),
        "brier_score_loss": d.get("brier_score_loss"),
        "n_samples": d.get("n_samples"),
        "results_path": str(p),
    }


def _delta(ours: float | None, theirs: float | None, gate: float) -> str:
    if ours is None or theirs is None:
        return "—"
    d = ours - theirs
    cell = f"{d:+.3f}"
    if abs(d) > gate:
        cell = f"**{cell}**"
    return cell


def _fmt(x: float | None) -> str:
    return f"{x:.3f}" if x is not None else "—"


def main() -> int:
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    rows: list[tuple[Combo, dict | None, dict | None]] = []
    for combo in all_combos(MODEL_ORDER):
        tf = _load_metrics(TF_DIR, combo)
        vl = _load_metrics(VL_DIR, combo)
        rows.append((combo, tf, vl))

    have_pairs = sum(1 for _, t, v in rows if t and v)
    only_tf = sum(1 for _, t, v in rows if t and not v)
    only_vl = sum(1 for _, t, v in rows if not t and v)

    lines: list[str] = []
    lines.append("# Table 1: vLLM backend vs transformers backend")
    lines.append("")
    lines.append(
        "Reads results from `results/paper-reproduction/` (transformers) and "
        "`results/paper-reproduction-vllm/` (vLLM). For each cell, reports the "
        "transformers AUC/ECE/Brier, the vLLM AUC/ECE/Brier, and Δ = vllm − transformers. "
        f"Cells with `|ΔAUC|>{GATE_AUC}` or `|ΔECE|>{GATE_ECE}` are **bolded** as gate failures."
    )
    lines.append("")
    lines.append(
        f"- Coverage: {have_pairs} matched pairs, {only_tf} transformers-only, "
        f"{only_vl} vLLM-only.\n"
        f"- Task: `{TASK}` | subsampling: {SUBSAMPLING} | seed: {SEED} | batch size: {BATCH_SIZE}\n"
    )

    metric_specs = [("AUC", "roc_auc", GATE_AUC), ("ECE", "ece", GATE_ECE), ("Brier", "brier_score_loss", GATE_ECE)]
    for title, key, gate in metric_specs:
        lines.append(f"## {title}")
        lines.append("")
        lines.append(
            "| Model | mode | transformers | vLLM | Δ |"
        )
        lines.append("|---|---|---|---|---|")
        for combo, tf, vl in rows:
            if tf is None and vl is None:
                continue
            mode_tag = "numeric" if combo.numeric else "MCQ"
            if combo.use_chat:
                mode_tag = "chat-" + mode_tag
            tf_v = (tf or {}).get(key)
            vl_v = (vl or {}).get(key)
            lines.append(
                f"| `{combo.model_name}` | {mode_tag} | {_fmt(tf_v)} | {_fmt(vl_v)} | {_delta(vl_v, tf_v, gate)} |"
            )
        lines.append("")

    OUT_PATH.write_text("\n".join(lines) + "\n")
    print(f"Wrote -> {OUT_PATH}")
    print(f"  matched pairs : {have_pairs}")
    print(f"  transformers only: {only_tf}")
    print(f"  vLLM only      : {only_vl}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
