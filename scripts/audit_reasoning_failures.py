"""Reasoning failure-rate audit (Phase 5).

For every reasoning-mode result in `results/extended-sweep/{transformers,vllm}`,
count the fraction of rows where the regex extractor fell back to 0.5
(`ReasoningQA.extract_probability_from_text`'s sentinel) and compare the
two backends.

A literal `0.5` is the failure marker — real MC/Numeric path scores cluster
around the model's mean confidence; landing on exactly 0.5 (within 1e-12)
is essentially deterministic-only when the regex returned None.

Usage:
    python scripts/audit_reasoning_failures.py
    python scripts/audit_reasoning_failures.py --root results/extended-sweep
    python scripts/audit_reasoning_failures.py --gate-pp 5     # acceptance gate (pp = percentage points)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent

# Modes considered "reasoning". Only these go through ReasoningQA.
REASONING_MODES = {"reasoning", "reasoning_think", "reasoning_thinking"}

# Threshold for matching the 0.5 fallback sentinel.
EPS = 1e-12


def _failure_rate(csv_path: Path) -> tuple[int, int, float]:
    """Return (n_total, n_failures, rate) for a predictions CSV."""
    df = pd.read_csv(csv_path)
    if "risk_score" not in df.columns:
        raise ValueError(f"CSV {csv_path} has no risk_score column")
    n = len(df)
    fail = int((df["risk_score"].sub(0.5).abs() < EPS).sum())
    return n, fail, fail / n if n else float("nan")


def _walk_results(root: Path) -> list[tuple[Path, dict]]:
    """Yield (results_json_path, parsed_dict) under `root`."""
    out: list[tuple[Path, dict]] = []
    for p in root.glob("**/results.bench-*.json"):
        try:
            with open(p) as f:
                d = json.load(f)
        except Exception:
            continue
        out.append((p, d))
    return out


def _is_reasoning(d: dict) -> bool:
    cfg = d.get("config", {})
    return bool(cfg.get("reasoning_prompting") or cfg.get("enable_thinking"))


def _model_name_from_path(p: Path) -> str:
    s = str(p)
    if "/model-" in s:
        return s.split("/model-")[1].split("_task-")[0]
    return p.parent.name


def _mode_tag(d: dict) -> str:
    cfg = d.get("config", {})
    if cfg.get("enable_thinking"):
        return "reasoning_think"
    if cfg.get("reasoning_prompting"):
        return "reasoning"
    if cfg.get("numeric_risk_prompting"):
        return "chat_numeric" if cfg.get("use_chat_template") else "numeric"
    return "chat_mcq" if cfg.get("use_chat_template") else "mcq"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", default="results/extended-sweep",
                        help="Sweep root containing transformers/ and vllm/ subdirs.")
    parser.add_argument("--gate-pp", type=float, default=5.0,
                        help="Acceptance gate: |Δ failure rate| ≤ gate-pp percentage points (default 5).")
    parser.add_argument("--out", default=None,
                        help="Output markdown path; default <root>/REASONING_FAILURE_AUDIT.md")
    args = parser.parse_args()

    root = (REPO_ROOT / args.root).resolve()
    if not root.exists():
        raise SystemExit(f"Sweep root does not exist: {root}")
    out_path = Path(args.out).resolve() if args.out else root / "REASONING_FAILURE_AUDIT.md"

    backends = [d.name for d in root.iterdir() if d.is_dir() and d.name in {"transformers", "vllm"}]
    if not backends:
        # Maybe results are flat (one backend per directory tree)
        backends = [""]

    rows: list[dict] = []
    for backend in backends:
        sub = root / backend if backend else root
        for results_json, d in _walk_results(sub):
            if not _is_reasoning(d):
                continue
            csv_rel = d.get("predictions_path")
            if not csv_rel:
                continue
            csv_path = Path(csv_rel)
            if not csv_path.exists():
                continue
            try:
                n, fail, rate = _failure_rate(csv_path)
            except Exception as exc:
                print(f"!! {results_json}: {exc}", file=sys.stderr)
                continue
            rows.append({
                "backend": backend,
                "model": _model_name_from_path(results_json),
                "mode": _mode_tag(d),
                "n_samples": n,
                "n_failures": fail,
                "fail_rate": rate,
                "auc": d.get("roc_auc"),
                "results_path": str(results_json),
            })

    if not rows:
        print("No reasoning results found under", root, file=sys.stderr)
        return 1

    df = pd.DataFrame(rows)

    # Build per-(model, mode) cross-backend pivot.
    pivot_rate = df.pivot_table(index=["model", "mode"], columns="backend",
                                values="fail_rate", aggfunc="first")
    pivot_n = df.pivot_table(index=["model", "mode"], columns="backend",
                             values="n_samples", aggfunc="first")

    lines: list[str] = []
    lines.append("# Reasoning failure-rate audit")
    lines.append("")
    lines.append(
        f"For each reasoning cell, the fraction of rows where "
        f"`risk_score == 0.5` (the regex-extraction fallback). "
        f"`Δ pp = vllm − transformers` percentage points. "
        f"Cells with `|Δ pp| > {args.gate_pp}` are **bolded**."
    )
    lines.append("")
    if "transformers" in pivot_rate.columns and "vllm" in pivot_rate.columns:
        lines.append("| Model | Mode | n | tf rate | vllm rate | Δ pp |")
        lines.append("|---|---|---|---|---|---|")
        for (model, mode), row in pivot_rate.iterrows():
            tf = row.get("transformers")
            vl = row.get("vllm")
            n = pivot_n.loc[(model, mode)].max() if (model, mode) in pivot_n.index else None
            if pd.isna(tf) and pd.isna(vl):
                continue
            tf_str = f"{tf:.3f}" if not pd.isna(tf) else "—"
            vl_str = f"{vl:.3f}" if not pd.isna(vl) else "—"
            if not pd.isna(tf) and not pd.isna(vl):
                d_pp = (vl - tf) * 100
                d_str = f"{d_pp:+.2f}"
                if abs(d_pp) > args.gate_pp:
                    d_str = f"**{d_str}**"
            else:
                d_str = "—"
            n_str = f"{int(n)}" if (n is not None and not pd.isna(n)) else "—"
            lines.append(f"| `{model}` | {mode} | {n_str} | {tf_str} | {vl_str} | {d_str} |")
    else:
        lines.append("(only one backend present in sweep root — cross-backend deltas not computed)")
        lines.append("")
        lines.append("| Model | Mode | backend | n | rate |")
        lines.append("|---|---|---|---|---|")
        for r in rows:
            lines.append(
                f"| `{r['model']}` | {r['mode']} | {r['backend']} "
                f"| {r['n_samples']} | {r['fail_rate']:.3f} |"
            )

    out_path.write_text("\n".join(lines) + "\n")
    print(f"Wrote -> {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
