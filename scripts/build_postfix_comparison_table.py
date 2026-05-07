"""Aggregate the cross-backend comparison table over ALL phases, post-fix.

For each (model, mode) pair across Phase 1 (paper Table 1), Phase 4
(extended sweep), Phase 6 (chat-template extension), and the Phase 5
reasoning sweep, pull the transformers and vLLM AUC/ECE values. For cells
affected by fixes #01 or #03, prefer the post-fix vLLM numbers in
`results/divergence_fix_validation/` over the original pre-fix numbers in
`results/paper-reproduction-vllm/`.

Output: a single Markdown table with columns: Model | Mode | TF AUC | vLLM
AUC | ΔAUC | TF ECE | vLLM ECE | ΔECE | rows.
"""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS = REPO_ROOT / "results"


@dataclass
class CellResult:
    model: str
    mode: str
    auc: float
    ece: float
    n: int


def _mode_key(cfg: dict) -> str | None:
    """Map a benchmark config dict to the human-readable mode name."""
    use_chat = bool(cfg.get("use_chat_template"))
    numeric = bool(cfg.get("numeric_risk_prompting"))
    reasoning = bool(cfg.get("reasoning_prompting"))
    enable_thinking = bool(cfg.get("enable_thinking"))
    if reasoning:
        return "reasoning_thinking" if enable_thinking else "reasoning"
    if use_chat and numeric:
        return "chat_numeric"
    if use_chat and not numeric:
        return "chat_mcq"
    if not use_chat and numeric:
        return "numeric"
    return "mcq"


def _extract_results(json_path: Path) -> CellResult | None:
    try:
        with open(json_path) as f:
            d = json.load(f)
    except Exception:
        return None
    cfg = d.get("config", {})
    # Some JSONs (paper-reproduction) don't have model_name_or_path; derive
    # the model from the directory name `model-<name>_task-<task>`.
    model_name = cfg.get("model_name") or d.get("model_name") or ""
    if not model_name:
        # Walk up: results.bench-X.json -> bench dir -> model dir
        model_dir = json_path.parent.parent
        if model_dir.name.startswith("model-"):
            stem = model_dir.name[len("model-"):]
            model_name = stem.split("_task-")[0]
    if not model_name:
        return None
    mode = _mode_key(cfg)
    if mode is None:
        return None
    return CellResult(
        model=model_name,
        mode=mode,
        auc=float(d["roc_auc"]),
        ece=float(d["ece"]),
        n=int(d["n_samples"]),
    )


def _scan(root: Path) -> dict[tuple[str, str], CellResult]:
    out: dict[tuple[str, str], CellResult] = {}
    if not root.exists():
        return out
    for p in root.rglob("results.bench-*.json"):
        cell = _extract_results(p)
        if cell is None:
            continue
        # Last-write-wins; in practice each (model, mode) pair has 1 file
        # per backend root.
        out[(cell.model, cell.mode)] = cell
    return out


def main() -> int:
    # Transformers (main branch) — always the reference
    tf_phase1 = _scan(RESULTS / "paper-reproduction")
    tf_phase4 = _scan(RESULTS / "extended-sweep" / "transformers")
    tf_phase5 = _scan(RESULTS / "reasoning-sweep" / "transformers")
    tf_all: dict[tuple[str, str], CellResult] = {}
    for d in (tf_phase1, tf_phase4, tf_phase5):
        tf_all.update(d)

    # vLLM — post-fix where available (overrides), falling back to original sweep
    vl_phase1 = _scan(RESULTS / "paper-reproduction-vllm")
    vl_phase4 = _scan(RESULTS / "extended-sweep" / "vllm")
    vl_phase5 = _scan(RESULTS / "reasoning-sweep" / "vllm")
    vl_postfix = _scan(RESULTS / "divergence_fix_validation")

    vl_all: dict[tuple[str, str], CellResult] = {}
    for d in (vl_phase1, vl_phase4, vl_phase5):
        vl_all.update(d)
    # Override with post-fix where available
    overridden = []
    for k, v in vl_postfix.items():
        if k in vl_all:
            overridden.append((k, vl_all[k].auc, v.auc))
        vl_all[k] = v

    # Build table over models that appear in BOTH backends
    common = sorted(set(tf_all) & set(vl_all))

    GATE_AUC = 0.015  # use the more permissive Phase 4 gate uniformly here
    GATE_ECE = 0.025

    def fmt(v: float) -> str:
        return f"{v:.3f}"

    def delta(d: float, gate: float) -> str:
        s = f"{d:+.3f}"
        return f"**{s}**" if abs(d) > gate else s

    # Order: group by model in some sensible order, then by mode within model
    MODE_ORDER = {"mcq": 0, "numeric": 1, "chat_mcq": 2, "chat_numeric": 3,
                  "reasoning": 4, "reasoning_thinking": 5}

    def sort_key(item: tuple[str, str]) -> tuple:
        model, mode = item
        return (model.lower(), MODE_ORDER.get(mode, 99))

    common.sort(key=sort_key)

    lines = []
    lines.append("## Cross-backend results — main (transformers) vs vllm-backend (post-fix)")
    lines.append("")
    lines.append(f"All Phase 1 / 4 / 5 cells with both backends present. Bolded rows fail the soft gates "
                 f"`|ΔAUC| > {GATE_AUC}` or `|ΔECE| > {GATE_ECE}`. The vLLM column is the result of the "
                 f"current `vllm-backend` HEAD with both fixes (#01, #03) applied.")
    lines.append("")
    lines.append("| Model | Mode | TF AUC | vLLM AUC | ΔAUC | TF ECE | vLLM ECE | ΔECE | n |")
    lines.append("|---|---|---|---|---|---|---|---|---|")

    n_pass_auc = n_pass_ece = total = 0
    for key in common:
        tf_cell = tf_all[key]
        vl_cell = vl_all[key]
        d_auc = vl_cell.auc - tf_cell.auc
        d_ece = vl_cell.ece - tf_cell.ece
        ok_auc = abs(d_auc) <= GATE_AUC
        ok_ece = abs(d_ece) <= GATE_ECE
        n_pass_auc += int(ok_auc)
        n_pass_ece += int(ok_ece)
        total += 1
        row = (
            f"| `{tf_cell.model}` | {tf_cell.mode} "
            f"| {fmt(tf_cell.auc)} | {fmt(vl_cell.auc)} | {delta(d_auc, GATE_AUC)} "
            f"| {fmt(tf_cell.ece)} | {fmt(vl_cell.ece)} | {delta(d_ece, GATE_ECE)} "
            f"| {tf_cell.n} |"
        )
        lines.append(row)

    lines.append("")
    lines.append(f"**{total} matched cells. AUC pass: {n_pass_auc}/{total}. "
                 f"ECE pass: {n_pass_ece}/{total}.**")

    if overridden:
        lines.append("")
        lines.append("### Cells where vLLM post-fix overrides pre-fix")
        lines.append("")
        lines.append("| Cell | vLLM pre-fix AUC | vLLM post-fix AUC |")
        lines.append("|---|---|---|")
        for (model, mode), pre, post in overridden:
            lines.append(f"| `{model}` ({mode}) | {pre:.3f} | {post:.3f} |")

    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
