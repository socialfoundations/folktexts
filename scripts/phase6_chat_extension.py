"""Phase 6: chat-template extension for Mistral-Instruct + Yi-Chat.

The default `reproduce_table1.CHAT_TEMPLATE_MODELS` only includes the two
Llama-3-Instruct variants. The Llama-3.2-1B chat-MCQ pre-flight showed a
systematic ECE shift of -0.04 (vLLM is *better* calibrated than transformers
on chat MC). To know whether this generalises beyond Llama, we re-run the
chat-MC and chat-numeric combos for Mistral-7B-Instruct-v0.2 and Yi-34B-Chat
on both backends.

This script monkey-patches `CHAT_TEMPLATE_MODELS` for its single invocation
so the main `reproduce_table1.py` is left untouched.

Usage:
    python scripts/phase6_chat_extension.py            # both backends
    python scripts/phase6_chat_extension.py --backends vllm
    python scripts/phase6_chat_extension.py --report-only
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import scripts.reproduce_table1 as repro  # noqa: E402

EXTRA_CHAT_MODELS = {
    "mistralai--Mistral-7B-Instruct-v0.2",
    "01-ai--Yi-34B-Chat",
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--backends", default="transformers,vllm",
                        help="Comma-separated; default 'transformers,vllm'.")
    parser.add_argument("--report-only", action="store_true",
                        help="Skip running; just rebuild the comparison report.")
    parser.add_argument("--no-skip", action="store_true",
                        help="Re-run cells even if matching JSON exists.")
    args = parser.parse_args()

    backends = [b.strip() for b in args.backends.split(",") if b.strip()]

    # Monkey-patch CHAT_TEMPLATE_MODELS to add the extra models for THIS run.
    repro.CHAT_TEMPLATE_MODELS = repro.CHAT_TEMPLATE_MODELS | EXTRA_CHAT_MODELS

    # Run only the two extra models, only their chat combos. Use all_combos +
    # filter to chat-only.
    selected = sorted(EXTRA_CHAT_MODELS)
    combos = [c for c in repro.all_combos(selected) if c.use_chat]
    print(f"Phase 6 — chat-template extension")
    print(f"Extra models: {selected}")
    print(f"Chat combos to run per backend: {[c.slug for c in combos]}\n")

    if args.report_only:
        # Just rerun the cross-backend comparison; the cells must already exist.
        import scripts.compare_table1_backends as cmp
        cmp.main()
        return 0

    fails = []
    for backend in backends:
        # Switch results dir per backend (mirrors reproduce_table1.main()).
        subdir = "paper-reproduction-vllm" if backend == "vllm" else "paper-reproduction"
        repro.RESULTS_DIR = REPO_ROOT / "results" / subdir
        repro.LOGS_DIR = repro.RESULTS_DIR / "logs"
        repro.LOGS_DIR.mkdir(parents=True, exist_ok=True)

        for c in combos:
            existing = repro._find_existing_result(c)
            if existing and not args.no_skip:
                print(f"-- {c.slug} ({backend}): skipping (existing)")
                continue
            try:
                repro.run_combo(c, backend=backend)
            except Exception as exc:
                import traceback
                tb = traceback.format_exc()
                log_path = repro.LOGS_DIR / f"{c.slug}__{backend}.log"
                log_path.write_text(tb)
                print(f"!! {c.slug} ({backend}) FAILED: {exc}; log -> {log_path}")
                fails.append(f"{c.slug}__{backend}")

    # Now run the cross-backend report
    import scripts.compare_table1_backends as cmp
    cmp.main()

    if fails:
        print(f"\n{len(fails)} cells failed: {fails}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
