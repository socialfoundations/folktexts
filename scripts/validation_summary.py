"""Aggregate validation artifacts into a single status doc for the PR.

Reads:
- results/paper-reproduction/TABLE1_REPRODUCTION.md          (TF vs paper)
- results/paper-reproduction-vllm/TABLE1_REPRODUCTION.md     (vLLM vs paper)
- results/paper-reproduction-vllm/TABLE1_BACKEND_COMPARISON.md (TF vs vLLM)
- results/multi-seed-stability/REPORT.md (if exists)
- results/extended-sweep/REPORT.md (if exists)
- results/extended-sweep/REASONING_FAILURE_AUDIT.md (if exists)

Writes results/VALIDATION_STATUS.md — a one-page overview with gate-pass
counts per phase, plus pointers to the underlying artifacts.

Usage:
    python scripts/validation_summary.py
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS = REPO_ROOT / "results"
OUT = RESULTS / "VALIDATION_STATUS.md"


def _try_read(p: Path) -> str:
    return p.read_text() if p.exists() else ""


def _count_gate_fails(report: str, section_name: str) -> tuple[int, int]:
    """Return (rows, fails) for a section in a backend-comparison-style report.

    A row is considered failed if the Δ column contains markdown bold (`**...**`)
    and both side columns are populated (no `—`).
    """
    if not report:
        return 0, 0
    parts = report.split(f"## {section_name}", 1)
    if len(parts) < 2:
        return 0, 0
    section = parts[1]
    section = re.split(r"^## ", section, maxsplit=1, flags=re.M)[0]
    rows = 0
    fails = 0
    for line in section.splitlines():
        if not line.startswith("| `"):
            continue
        cols = [c.strip() for c in line.split("|")]
        # cols layout: ['', model, mode, tf, vllm, delta, ''] or similar
        if len(cols) < 6:
            continue
        tf = cols[3] if len(cols) > 3 else ""
        vl = cols[4] if len(cols) > 4 else ""
        d = cols[5] if len(cols) > 5 else ""
        if tf == "—" or vl == "—":
            continue
        rows += 1
        if "**" in d:
            fails += 1
    return rows, fails


def main() -> int:
    tf_paper = _try_read(RESULTS / "paper-reproduction" / "TABLE1_REPRODUCTION.md")
    vl_paper = _try_read(RESULTS / "paper-reproduction-vllm" / "TABLE1_REPRODUCTION.md")
    cross = _try_read(RESULTS / "paper-reproduction-vllm" / "TABLE1_BACKEND_COMPARISON.md")
    multiseed = _try_read(RESULTS / "multi-seed-stability" / "REPORT.md")
    extended = _try_read(RESULTS / "extended-sweep" / "REPORT.md")
    reasoning = _try_read(RESULTS / "extended-sweep" / "REASONING_FAILURE_AUDIT.md")

    cross_rows_auc, cross_fails_auc = _count_gate_fails(cross, "AUC")
    cross_rows_ece, cross_fails_ece = _count_gate_fails(cross, "ECE")

    extended_rows_auc, extended_fails_auc = _count_gate_fails(extended, "AUC")
    extended_rows_ece, extended_fails_ece = _count_gate_fails(extended, "ECE")

    # Match cross-backend coverage from the comparison report header.
    coverage_match = re.search(
        r"Coverage:\s*(\d+)\s*matched pairs", cross or ""
    )
    matched_pairs = coverage_match.group(1) if coverage_match else "n/a"

    lines: list[str] = []
    lines.append("# vLLM migration — validation status")
    lines.append("")
    lines.append(
        "Aggregates the artifacts under `results/`. Each phase has a gate that "
        "the migration must pass before flipping the CLI default."
    )
    lines.append("")
    lines.append("## Phase 1 — Cross-backend Table 1 (vLLM vs transformers)")
    lines.append("")
    if cross:
        lines.append(f"- Source: `results/paper-reproduction-vllm/TABLE1_BACKEND_COMPARISON.md`")
        lines.append(f"- Coverage: **{matched_pairs}** matched pairs (target: 20)")
        lines.append(
            f"- AUC gate (|Δ| ≤ 0.01): "
            f"**{cross_rows_auc - cross_fails_auc}/{cross_rows_auc}** within tolerance"
        )
        lines.append(
            f"- ECE gate (|Δ| ≤ 0.02): "
            f"**{cross_rows_ece - cross_fails_ece}/{cross_rows_ece}** within tolerance"
        )
    else:
        lines.append("- _Not yet generated_ — run `scripts/compare_table1_backends.py`.")
    lines.append("")

    lines.append("## Phase 2 — Paper Table 1 reproduction")
    lines.append("")
    if tf_paper:
        lines.append("- transformers vs paper: `results/paper-reproduction/TABLE1_REPRODUCTION.md`")
    else:
        lines.append("- transformers vs paper: _missing_")
    if vl_paper:
        lines.append("- vLLM vs paper: `results/paper-reproduction-vllm/TABLE1_REPRODUCTION.md`")
    else:
        lines.append("- vLLM vs paper: _missing_ — run `scripts/reproduce_table1.py --backend vllm --report-only`")
    lines.append("")

    lines.append("## Phase 3 — Multi-seed stability")
    lines.append("")
    if multiseed:
        lines.append("- Source: `results/multi-seed-stability/REPORT.md`")
    else:
        lines.append("- _Not yet run_ — `python scripts/multi_seed_stability.py`")
    lines.append("")

    lines.append("## Phase 4 — Modern + thinking-model coverage")
    lines.append("")
    if extended:
        lines.append("- Source: `results/extended-sweep/REPORT.md`")
        lines.append(
            f"- AUC gate (|Δ| ≤ 0.015): "
            f"**{extended_rows_auc - extended_fails_auc}/{extended_rows_auc}** within tolerance"
        )
        lines.append(
            f"- ECE gate (|Δ| ≤ 0.025): "
            f"**{extended_rows_ece - extended_fails_ece}/{extended_rows_ece}** within tolerance"
        )
    else:
        lines.append("- _Not yet run_ — `python scripts/extended_sweep.py --tier tier1`")
    lines.append("")

    lines.append("## Phase 5 — Reasoning failure-rate audit")
    lines.append("")
    if reasoning:
        lines.append("- Source: `results/extended-sweep/REASONING_FAILURE_AUDIT.md`")
    else:
        lines.append(
            "- _Not yet run_ (depends on Phase 4) — `python scripts/audit_reasoning_failures.py`"
        )
    lines.append("")

    OUT.write_text("\n".join(lines) + "\n")
    print(f"Wrote -> {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
