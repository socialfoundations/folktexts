"""Compare pre-fix and post-fix vLLM Llama-3 numeric results vs transformers.

Reads the existing TF result and the new post-fix vLLM result, prints AUC/ECE
deltas and prediction distribution stats.
"""
from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

import pandas as pd


def _summarise(json_path: Path, label: str) -> dict:
    j = json.load(open(json_path))
    csv = glob.glob(str(json_path.parent) + "/*test_predictions.csv")[0]
    df = pd.read_csv(csv)
    rs = df["risk_score"]
    return {
        "label": label,
        "auc": j["roc_auc"],
        "ece": j["ece"],
        "brier": j["brier_score_loss"],
        "unique": rs.nunique(),
        "mean": rs.mean(),
        "std": rs.std(),
        "top5": dict(rs.value_counts().head()),
    }


def main() -> int:
    results = []
    for label, glob_pattern in [
        ("TF (paper-reproduction)",
         "/lustre/home/acruz/folktexts/results/paper-reproduction/model-meta-llama--Meta-Llama-3-8B_task-ACSIncome/*/results.bench-*.json"),
        ("vLLM PRE-fix",
         "/lustre/home/acruz/folktexts/results/paper-reproduction-vllm/model-meta-llama--Meta-Llama-3-8B_task-ACSIncome/*/results.bench-*.json"),
        ("vLLM POST-fix",
         "/lustre/home/acruz/folktexts/results/divergence_fix_validation/llama3-8b-numeric-vllm-fixed/model-meta-llama--Meta-Llama-3-8B_task-ACSIncome/*/results.bench-*.json"),
    ]:
        for jp in glob.glob(glob_pattern):
            try:
                j = json.load(open(jp))
            except Exception as e:
                continue
            cfg = j.get("config", {})
            if cfg.get("numeric_risk_prompting") and not cfg.get("use_chat_template"):
                results.append(_summarise(Path(jp), label))

    if not results:
        print("No matching JSONs found.")
        return 1

    print("\n" + "=" * 100)
    print(f"{'Label':<40s} {'AUC':>8s} {'ECE':>8s} {'Brier':>8s} {'Unique':>8s} {'Mean':>8s} {'Std':>8s}")
    print("=" * 100)
    for r in results:
        print(f"{r['label']:<40s} {r['auc']:>8.4f} {r['ece']:>8.4f} {r['brier']:>8.4f} "
              f"{r['unique']:>8d} {r['mean']:>8.4f} {r['std']:>8.4f}")
    print("=" * 100)
    for r in results:
        print(f"{r['label']:<40s} top-5 risk_scores:")
        for k, v in list(r["top5"].items())[:5]:
            print(f"    {k}: {v}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
