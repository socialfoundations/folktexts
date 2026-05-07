"""Probe how the unmasked next-token distribution depends on prefill wording.

For Llama-3-8B base on a real ACSIncome prompt body, vary only the trailing
"answer suffix" and dump:
  - pos-0 top-15 tokens (the next token the model wants after the prefill)
  - total probability mass on digits ("0".."9") at pos-0
  - if pos-0 is digit-forced, what pos-1 looks like (top-15)

This isolates whether the suffix is the lever — we want a suffix where digits
naturally dominate pos-0 AND the 2nd-position continuation is also digit-heavy
(so the 2-forward-pass decoder picks a meaningful "second decimal").

We use vLLM `raw_logprobs` so we see the un-masked distribution.

The body of the prompt is shared across suffix variants — we only swap the
last few characters, so any difference is attributable to the suffix.
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

MODEL_PATH = "/fast/groups/sf/huggingface-models/meta-llama--Meta-Llama-3-8B"
DATA_DIR = "/fast/acruz/data/folktables"
N_ROWS = 8


@dataclass(frozen=True)
class Variant:
    label: str
    suffix: str  # replaces the question-line trailing "Answer..." part


VARIANTS = [
    Variant("baseline_current",         "\nAnswer (between 0 and 1): 0."),
    Variant("baseline_no_paren",        "\nAnswer: 0."),
    Variant("probability_word",         "\nProbability: 0."),
    Variant("explicit_4dp_hint",        "\nProbability (4 decimal places): 0."),
    Variant("p_equals",                 "\nP = 0."),
    Variant("yes_probability",          '\nProbability of "yes": 0.'),
    Variant("between_no_prefill",       "\nAnswer (between 0 and 1): "),
    Variant("decimal_lead_zero",        "\nThe probability is 0."),
    Variant("with_example",             "\nAnswer (e.g., 0.6789, between 0 and 1): 0."),
    Variant("range_in_word_form",       "\nAnswer between zero and one: 0."),
    Variant("strict_format",            "\nReply in the form 0.XXXX. Answer: 0."),
    Variant("only_digit_after_zero",    "\nAnswer (between 0.0001 and 0.9999): 0."),
]


def make_body(task, row, question_text: str) -> str:
    from folktexts.prompting import ACS_TASK_DESCRIPTION
    return (
        ACS_TASK_DESCRIPTION + "\n"
        + f"\nInformation:\n{task.get_row_description(row)}\n"
        + f"\nQuestion: {question_text}"
    )


def main() -> int:
    os.environ.setdefault("VLLM_USE_DEEP_GEMM", "0")
    os.environ.setdefault("VLLM_LOGGING_LEVEL", "WARNING")

    from transformers import AutoConfig, AutoTokenizer

    from folktexts.acs.acs_columns import acs_income_numeric_qa
    from folktexts.acs.acs_dataset import ACSDataset
    from folktexts.acs.acs_tasks import ACSTaskMetadata
    from folktexts.task import TaskMetadata

    task = TaskMetadata.get_task("ACSIncome")
    if not isinstance(task, ACSTaskMetadata):
        print(f"Unexpected task type: {type(task).__name__}")
        return 1
    dataset = ACSDataset.make_from_task(
        task=task, cache_dir=DATA_DIR, subsampling=0.005, seed=42,
    )
    rows = dataset.data.iloc[:N_ROWS]
    question_text = acs_income_numeric_qa.text

    cfg = AutoConfig.from_pretrained(MODEL_PATH)
    vocab_dim = int(cfg.vocab_size)
    tok = AutoTokenizer.from_pretrained(MODEL_PATH)

    digit_token_ids = sorted({
        tid for tk, tid in tok.get_vocab().items()
        if tk.isdecimal() and 0 <= tid < vocab_dim
    })
    digit_id_set = set(digit_token_ids)
    print(f"vocab_dim={vocab_dim}  decimal_token_count={len(digit_token_ids)}")

    # Construct full prompts: per row, per variant.
    bodies = [make_body(task, row, question_text) for _, row in rows.iterrows()]

    prompts: list[str] = []
    prompt_meta: list[tuple[str, int]] = []  # (variant_label, row_idx)
    for variant in VARIANTS:
        for i, body in enumerate(bodies):
            prompts.append(body + variant.suffix)
            prompt_meta.append((variant.label, i))

    # Sanity: print the last 80 chars of each variant on row 0 just so we know
    # what we're feeding the model.
    print("\n--- Variant suffix samples (row 0 tail) ---")
    for variant in VARIANTS:
        idx = next(j for j, m in enumerate(prompt_meta) if m == (variant.label, 0))
        print(f"  [{variant.label}] tail = {prompts[idx][-80:]!r}")

    from vllm import LLM, SamplingParams

    llm = LLM(
        model=MODEL_PATH,
        dtype="bfloat16",
        gpu_memory_utilization=0.6,
        max_model_len=2048,
        max_logprobs=200,
        seed=42,
        logprobs_mode="raw_logprobs",  # we want the UNMASKED distribution
    )

    sampling = SamplingParams(
        temperature=0.0,
        max_tokens=2,
        logprobs=50,
        # NOTE: deliberately no allowed_token_ids — we want raw, free generation
        seed=42,
    )

    outputs = llm.generate(prompts, sampling)

    # Aggregate: per variant, average across N_ROWS rows.
    import math

    @dataclass
    class Agg:
        digit_mass_pos0: float = 0.0
        digit_mass_pos1: float = 0.0
        period_mass_pos0: float = 0.0
        newline_mass_pos0: float = 0.0
        eot_mass_pos0: float = 0.0
        n_rows: int = 0

    aggs: dict[str, Agg] = {v.label: Agg() for v in VARIANTS}

    # We'll also dump the full top-15 of pos-0 and pos-1 for ROW 0 of each
    # variant, so we can read them by eye.
    detail_lines: list[str] = []

    NEWLINE_ID = tok.encode("\n", add_special_tokens=False)
    PERIOD_ID = 13                  # Llama-3 '.'
    EOT_ID = tok.eos_token_id

    for prompt, (label, row_idx), out in zip(prompts, prompt_meta, outputs):
        completion = out.outputs[0]
        agg = aggs[label]
        agg.n_rows += 1

        if not completion.logprobs:
            continue

        for pos_idx, pos in enumerate(completion.logprobs[:2]):
            sorted_pos = sorted(
                ((int(tid), getattr(lp, "decoded_token", None), float(getattr(lp, "logprob", lp)))
                 for tid, lp in pos.items()),
                key=lambda x: x[2], reverse=True,
            )

            digit_mass = 0.0
            period_mass = 0.0
            newline_mass = 0.0
            eot_mass = 0.0
            for tid, _decoded, logp in sorted_pos:
                p = math.exp(logp)
                if tid in digit_id_set:
                    digit_mass += p
                if tid == PERIOD_ID:
                    period_mass += p
                if tid in NEWLINE_ID:
                    newline_mass += p
                if tid == EOT_ID:
                    eot_mass += p

            if pos_idx == 0:
                agg.digit_mass_pos0 += digit_mass
                agg.period_mass_pos0 += period_mass
                agg.newline_mass_pos0 += newline_mass
                agg.eot_mass_pos0 += eot_mass
            else:
                agg.digit_mass_pos1 += digit_mass

            if row_idx == 0 and pos_idx <= 1:
                detail_lines.append(
                    f"\n[{label}] pos={pos_idx}  digit_mass={digit_mass:.4f}  "
                    f"period={period_mass:.4f}  newline={newline_mass:.4f}  eot={eot_mass:.4f}"
                )
                for tid, decoded, logp in sorted_pos[:15]:
                    p = math.exp(logp)
                    is_digit = "D" if tid in digit_id_set else " "
                    detail_lines.append(
                        f"     {is_digit} id={tid:6d}  p={p:.4f}  decoded={decoded!r}"
                    )

    # Print detail (per variant, row 0)
    print("\n=== Detailed top-15 (row 0 only, pos-0 and pos-1) ===")
    print("\n".join(detail_lines))

    print("\n=== Aggregate over {} rows ===".format(N_ROWS))
    print(
        "  variant".ljust(36),
        "digit@0".rjust(10),
        "digit@1".rjust(10),
        "'.'@0".rjust(8),
        "'\\n'@0".rjust(8),
        "EOT@0".rjust(8),
    )
    for variant in VARIANTS:
        a = aggs[variant.label]
        n = max(a.n_rows, 1)
        print(
            f"  {variant.label}".ljust(36),
            f"{a.digit_mass_pos0/n:.3f}".rjust(10),
            f"{a.digit_mass_pos1/n:.3f}".rjust(10),
            f"{a.period_mass_pos0/n:.3f}".rjust(8),
            f"{a.newline_mass_pos0/n:.3f}".rjust(8),
            f"{a.eot_mass_pos0/n:.3f}".rjust(8),
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
