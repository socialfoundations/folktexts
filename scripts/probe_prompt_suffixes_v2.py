"""V2: probe pos-1 with pos-0 FORCED to a digit.

The original finding was: "for DirectNumericQA on Llama-3, the unmasked pos-1
distribution is dominated by '\\n', <|end_of_text|>, and '.'". That finding is
about what the model WANTS at pos-1 *after* it has been forced to emit a digit
at pos-0 (production behaviour with `allowed_token_ids=digit_ids`).

This probe replicates that exactly:
  - SamplingParams: temperature=0, max_tokens=2, allowed_token_ids=digits,
    logprobs=50, logprobs_mode="raw_logprobs"  (KEY: raw, so we see the
    UNMASKED distribution even when the mask is in effect).
  - For each prompt variant, aggregate over N rows:
      * pos-0 digit mass (will be 1.0 because of mask-forced selection;
        but we also report what the unmasked digit mass would have been —
        i.e. how naturally the model wanted a digit at pos-0).
      * pos-1 unmasked digit mass — THE key number. If low, model thought
        the answer was complete after pos-0.
      * pos-1 mass on '\\n', <|eot|>, '.' specifically.
"""
from __future__ import annotations

import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

MODEL_PATH = "/fast/groups/sf/huggingface-models/meta-llama--Meta-Llama-3-8B"
DATA_DIR = "/fast/acruz/data/folktables"
N_ROWS = 16


@dataclass(frozen=True)
class Variant:
    label: str
    suffix: str


VARIANTS = [
    Variant("baseline_current",    "\nAnswer (between 0 and 1): 0."),
    Variant("baseline_no_paren",   "\nAnswer: 0."),
    Variant("probability_word",    "\nProbability: 0."),
    Variant("decimal_lead_zero",   "\nThe probability is 0."),
    Variant("strict_format",       "\nReply in the form 0.XXXX. Answer: 0."),
    Variant("answer_zero_only",    "\nAnswer: 0"),  # no trailing period
    Variant("answer_with_decimal", "\nAnswer: 0.5"),  # 1-digit prefill, model picks 2nd
    Variant("p_equals",            "\nP = 0."),
    Variant("estimate_is",         "\nThe estimated probability is 0."),
    Variant("specifically",        "\nSpecifically, the probability is 0."),
    Variant("rounded_to_two",      "\nRounded to two decimals, the probability is 0."),
    Variant("digit_after_zero",    "\nAnswer: 0.5"),  # baseline-ish for compare
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

    bodies = [make_body(task, row, question_text) for _, row in rows.iterrows()]
    prompts: list[str] = []
    prompt_meta: list[tuple[str, int]] = []
    for variant in VARIANTS:
        for i, body in enumerate(bodies):
            prompts.append(body + variant.suffix)
            prompt_meta.append((variant.label, i))

    from vllm import LLM, SamplingParams

    llm = LLM(
        model=MODEL_PATH,
        dtype="bfloat16",
        gpu_memory_utilization=0.6,
        max_model_len=2048,
        max_logprobs=200,
        seed=42,
        logprobs_mode="raw_logprobs",  # see UNMASKED distribution
    )

    sampling = SamplingParams(
        temperature=0.0,
        max_tokens=2,
        logprobs=50,
        allowed_token_ids=digit_token_ids,  # production: force pos-0 and pos-1 to be digits
        seed=42,
    )

    outputs = llm.generate(prompts, sampling)

    NEWLINE_IDS = set(tok.encode("\n", add_special_tokens=False))
    EOT_ID = tok.eos_token_id
    PERIOD_ID = 13

    @dataclass
    class Agg:
        digit_pos0: float = 0.0
        digit_pos1: float = 0.0
        period_pos1: float = 0.0
        newline_pos1: float = 0.0
        eot_pos1: float = 0.0
        n: int = 0

    aggs: dict[str, Agg] = {v.label: Agg() for v in VARIANTS}
    detail_lines: list[str] = []

    for prompt, (label, row_idx), out in zip(prompts, prompt_meta, outputs):
        completion = out.outputs[0]
        agg = aggs[label]
        agg.n += 1

        if not completion.logprobs:
            continue

        for pos_idx, pos in enumerate(completion.logprobs[:2]):
            sorted_pos = sorted(
                ((int(tid), getattr(lp, "decoded_token", None), float(getattr(lp, "logprob", lp)))
                 for tid, lp in pos.items()),
                key=lambda x: x[2], reverse=True,
            )

            digit = period = newline = eot = 0.0
            for tid, _decoded, logp in sorted_pos:
                p = math.exp(logp)
                if tid in digit_id_set:
                    digit += p
                if tid == PERIOD_ID:
                    period += p
                if tid in NEWLINE_IDS:
                    newline += p
                if tid == EOT_ID:
                    eot += p

            if pos_idx == 0:
                agg.digit_pos0 += digit
            else:
                agg.digit_pos1 += digit
                agg.period_pos1 += period
                agg.newline_pos1 += newline
                agg.eot_pos1 += eot

            if row_idx == 0 and pos_idx == 1:
                detail_lines.append(
                    f"\n[{label}] pos=1  unmasked: digit={digit:.4f}  '.'={period:.4f}  "
                    f"'\\n'={newline:.4f}  EOT={eot:.4f}"
                )
                for tid, decoded, logp in sorted_pos[:12]:
                    p = math.exp(logp)
                    is_digit = "D" if tid in digit_id_set else " "
                    detail_lines.append(
                        f"     {is_digit} id={tid:6d}  p={p:.4f}  decoded={decoded!r}"
                    )

    print("\n=== POS-1 detailed top-12 (row 0, unmasked dist with pos-0 forced to digit) ===")
    print("\n".join(detail_lines))

    print("\n=== Aggregate over {} rows (raw_logprobs, mask ON) ===".format(N_ROWS))
    print(
        "  variant".ljust(32),
        "ud@0".rjust(8),
        "ud@1".rjust(8),
        "'.'@1".rjust(8),
        "'\\n'@1".rjust(8),
        "EOT@1".rjust(8),
        "non-digit@1".rjust(12),
    )
    for variant in VARIANTS:
        a = aggs[variant.label]
        n = max(a.n, 1)
        ud0 = a.digit_pos0 / n
        ud1 = a.digit_pos1 / n
        per = a.period_pos1 / n
        nl = a.newline_pos1 / n
        eot = a.eot_pos1 / n
        non_digit = per + nl + eot
        print(
            f"  {variant.label}".ljust(32),
            f"{ud0:.3f}".rjust(8),
            f"{ud1:.3f}".rjust(8),
            f"{per:.3f}".rjust(8),
            f"{nl:.3f}".rjust(8),
            f"{eot:.3f}".rjust(8),
            f"{non_digit:.3f}".rjust(12),
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
