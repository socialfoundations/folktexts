#!/usr/bin/env python3
"""Test script for the new ReasoningQA functionality.

This script tests chain-of-thought reasoning prompting on a small subsample
of the ACSIncome dataset.

Usage:
    python scripts/test_reasoning_qa.py --model <model_name_or_path> [--enable-thinking]

Examples:
    # Test with a local model (standard reasoning mode)
    python scripts/test_reasoning_qa.py --model models/google--gemma-2b

    # Test with Qwen3 thinking mode
    python scripts/test_reasoning_qa.py --model Qwen/Qwen3-32B --enable-thinking

    # Test with a smaller model for quick debugging
    python scripts/test_reasoning_qa.py --model gpt2

    # Quick single-sample test to debug generated output
    python scripts/test_reasoning_qa.py --model Qwen/Qwen3-4B --test-single
"""
import argparse
import logging
import os
import sys
from pathlib import Path

# Add the parent directory to the path so we can import folktexts
sys.path.insert(0, str(Path(__file__).parent.parent))


def setup_logging(level: str = "INFO"):
    """Configure logging for the test script."""
    logging.basicConfig(
        level=getattr(logging, level),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


def test_probability_extraction():
    """Test the probability extraction patterns work correctly."""
    from folktexts.qa_interface import ReasoningQA

    test_cases = [
        # (input_text, expected_probability)
        # Standard formats
        ("After considering all factors, Probability: 75%", 0.75),
        ("Based on the analysis, Probability: 0.65", 0.65),
        ("The probability is Probability: 80 percent", 0.80),
        ("I estimate about 45%", 0.45),
        ("Final answer: 0.32", 0.32),
        ("probability: 60%", 0.60),
        ("Probability: 25%\n", 0.25),
        ("thinking... Probability: 0.90", 0.90),

        # Variations on "Probability" format
        ("The probability estimate is 55%", 0.55),
        ("Probability of success: 0.70", 0.70),
        ("probability is 0.85", 0.85),

        # Multiple percentages (should use LAST one)
        ("First I thought 30%, but actually Probability: 70%", 0.70),
        ("The base rate is 50%, adjusted to 65%", 0.65),

        # Thinking model output with probability in response (correct format)
        ("<think>Analyzing factors...</think>\n\nProbability: 80%", 0.80),

        # Note: extract_probability_from_text doesn't know about <think> tags.
        # The separation happens in generate_text_batch, which passes only the
        # response content (after </think>) to extraction. These tests verify
        # the raw extraction works on whatever text is passed.
        ("<think>Based on factors, Probability: 65%</think>\n\nMy final answer.", 0.65),

        # This simulates what the extraction sees when thinking is properly separated:
        # Only the response content is passed (no <think> tags)
        ("Based on the analysis above, my estimate is:\n\nProbability: 72%", 0.72),

        # Empty or no-probability response (would fail in pipeline)
        ("My final answer.", None),

        # Edge cases - should NOT extract
        ("No probability here", None),
        ("The value is 150 which is not a probability", None),
        ("The year is 2018", None),
    ]

    print("\n" + "=" * 60)
    print("Testing probability extraction patterns")
    print("=" * 60)

    all_passed = True
    for text, expected in test_cases:
        result = ReasoningQA.extract_probability_from_text(text)

        # For comparison, handle floating point precision
        if result is not None and expected is not None:
            passed = abs(result - expected) < 0.001
        else:
            passed = result == expected

        status = "✓" if passed else "✗"
        if not passed:
            all_passed = False
            print(f"{status} Input: {text[:60]!r}...")
            print(f"   Expected: {expected}, Got: {result}")
        else:
            display_text = text[:50] + "..." if len(text) > 50 else text
            print(f"{status} {display_text!r} -> {result}")

    print("=" * 60)
    print(f"All tests passed: {all_passed}")
    return all_passed


def test_single_sample(
    model,
    tokenizer,
    *,
    data_dir: str,
    enable_thinking: bool = False,
    num_samples: int = 3,
    log_generations: bool = False,
    log_generations_first_n: int = 3,
):
    """Test generation on a few samples to debug output format.

    Parameters
    ----------
    model : AutoModelForCausalLM
        The loaded model.
    tokenizer : AutoTokenizer
        The loaded tokenizer.
    data_dir : str
        Root folder to find/download datasets (ACS folktables cache will live under
        ``{data_dir}/folktables``).
    enable_thinking : bool
        Whether to enable thinking mode.
    num_samples : int
        Number of samples to test.
    log_generations : bool
        If True, log all prompts and generations fully. If False (default),
        log only the first ``log_generations_first_n`` prompts/generations fully.
    log_generations_first_n : int
        Number of prompts/generations to log fully when ``log_generations=False``.
    """
    from folktexts.acs.acs_dataset import ACSDataset
    from folktexts.acs.acs_tasks import ACSTaskMetadata
    from folktexts.llm_utils import generate_text_batch
    from folktexts.prompting import encode_row_prompt
    from folktexts.qa_interface import ReasoningQA

    logger = logging.getLogger(__name__)

    logger.info("=" * 60)
    logger.info("Testing single sample generation")
    logger.info("=" * 60)

    # Get the ACS task
    task = ACSTaskMetadata.get_task(name="ACSIncome", use_numeric_qa=False)

    # Create reasoning QA
    base_question = task.direct_numeric_qa
    reasoning_qa = ReasoningQA(
        column=base_question.column,
        text=base_question.text,
        enable_thinking=enable_thinking,
    )
    task.set_question(reasoning_qa)

    logger.info("Question configuration:")
    logger.info(f"  - enable_thinking: {reasoning_qa.enable_thinking}")
    logger.info(f"  - max_new_tokens: {reasoning_qa.max_new_tokens}")
    logger.info(f"  - Question text: {reasoning_qa.text}")

    # Check if tokenizer supports enable_thinking
    logger.info("Tokenizer info:")
    logger.info(f"  - Name: {tokenizer.name_or_path}")
    logger.info(f"  - Has chat_template: {hasattr(tokenizer, 'chat_template') and tokenizer.chat_template is not None}")

    # Load a small sample of data
    logger.info(f"Loading dataset (data_dir={data_dir})...")
    acs_dataset = ACSDataset.make_from_task(
        task=task,
        cache_dir=data_dir,
        survey_year="2018",
        horizon="1-Year",
        survey="person",
        test_size=0.1,
        val_size=0.1,
        subsampling=0.01,  # Small for testing
        seed=42,
    )

    X_test, y_test = acs_dataset.get_test()
    logger.info(f"Test set size: {len(X_test)}")

    results = []

    n_to_run = min(num_samples, len(X_test))
    for i in range(n_to_run):
        logger.info("=" * 60)
        logger.info(f"SAMPLE {i + 1}/{n_to_run}")
        logger.info("=" * 60)

        # Get sample
        sample_row = X_test.iloc[i]
        sample_label = y_test.iloc[i]

        # Encode the prompt
        prompt = encode_row_prompt(sample_row, task=task, question=reasoning_qa)

        should_log_full = log_generations or (i < max(log_generations_first_n, 0))
        if should_log_full:
            logger.info("-" * 60)
            logger.info("MODEL INPUT PROMPT:")
            logger.info("-" * 60)
            logger.info(prompt)
            logger.info("-" * 60)

        # Generate text
        logger.info("Generating text...")

        generated_texts = generate_text_batch(
            text_inputs=[prompt],
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=1000,  # Reasonable length for reasoning
            context_size=2048,
            enable_thinking=enable_thinking,
        )

        generated_text = generated_texts[0]

        if should_log_full:
            logger.info("-" * 60)
            logger.info(f"MODEL GENERATED OUTPUT ({len(generated_text)} chars):")
            logger.info("-" * 60)
            logger.info(generated_text)
            logger.info("-" * 60)

        # Extract probability
        extracted_prob = reasoning_qa.get_answer_from_model_output(generated_text)

        logger.info(f"Extracted probability: {extracted_prob:.6f}")
        logger.info(f"True label: {sample_label}")

        # Check if extraction might have failed (returns 0.5 default)
        if extracted_prob == 0.5:
            logger.info("WARNING: Extracted probability is exactly 0.5 - extraction may have failed!")

        results.append({
            "sample_idx": i,
            "generated_text": generated_text,
            "extracted_probability": extracted_prob,
            "true_label": sample_label,
        })

    # Summary
    logger.info("=" * 60)
    logger.info("SUMMARY")
    logger.info("=" * 60)

    probs = [r["extracted_probability"] for r in results]
    labels = [r["true_label"] for r in results]

    logger.info(f"Extracted probabilities: {probs}")
    logger.info(f"True labels: {labels}")

    # Check for concerning patterns
    if all(p == 0.5 for p in probs):
        logger.info("❌ ALL probabilities are 0.5 - extraction is likely failing!")
    elif len(set(probs)) == 1:
        logger.info(f"⚠️  All probabilities are identical ({probs[0]}) - may indicate a problem")
    else:
        logger.info(f"✓ Probabilities have spread from (min={min(probs):.2f}) to (max={max(probs):.2f})")

    return results


def run_full_benchmark(args):
    """Run the full benchmark test."""
    logger = logging.getLogger(__name__)

    logger.info("=" * 60)
    logger.info("Testing ReasoningQA functionality")
    logger.info("=" * 60)
    logger.info(f"Model: {args.model}")
    logger.info(f"Enable thinking: {args.enable_thinking}")
    logger.info(f"Subsampling: {args.subsampling:.1%}")
    logger.info(f"Batch size: {args.batch_size}")
    logger.info("=" * 60)

    # Load model and tokenizer
    logger.info("Loading model and tokenizer...")
    from folktexts.llm_utils import load_model_tokenizer
    model, tokenizer = load_model_tokenizer(args.model)
    logger.info(f"Model loaded: {model.name_or_path}")

    # Create benchmark config with reasoning prompting
    from folktexts.benchmark import Benchmark, BenchmarkConfig

    config = BenchmarkConfig(
        reasoning_prompting=True,
        enable_thinking=args.enable_thinking,
        batch_size=args.batch_size,
        correct_order_bias=False,  # Not applicable for ReasoningQA
    )
    logger.info(f"Benchmark config: {config}")

    # Create the benchmark
    logger.info("Creating ACS benchmark...")
    bench = Benchmark.make_acs_benchmark(
        task_name="ACSIncome",
        model=model,
        tokenizer=tokenizer,
        data_dir=args.data_dir,
        config=config,
        subsampling=args.subsampling,
    )

    # Log task question info
    task = bench.task
    logger.info(f"Task: {task.name}")
    logger.info(f"Question type: {type(task.question).__name__}")
    logger.info(f"Question text: {task.question.text}")
    if hasattr(task.question, 'enable_thinking'):
        logger.info(f"Enable thinking: {task.question.enable_thinking}")
    if hasattr(task.question, 'max_new_tokens'):
        logger.info(f"Max new tokens: {task.question.max_new_tokens}")

    # Show example prompt
    X_test, y_test = bench.dataset.get_test()
    example_row = X_test.iloc[0]
    example_prompt = bench.llm_clf.encode_row(example_row, question=task.question)
    logger.info("=" * 60)
    logger.info("Example prompt:")
    logger.info("=" * 60)
    print(example_prompt)
    logger.info("=" * 60)

    # Create results directory
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    # Run the benchmark
    logger.info("Running benchmark...")
    results = bench.run(results_root_dir=results_dir)

    # Print results summary
    logger.info("=" * 60)
    logger.info("Results Summary")
    logger.info("=" * 60)
    logger.info(f"ECE: {results['ece']:.2%}")
    logger.info(f"ROC AUC: {results['roc_auc']:.2%}")
    logger.info(f"Accuracy: {results['accuracy']:.2%}")
    logger.info(f"Balanced Accuracy: {results['balanced_accuracy']:.2%}")
    logger.info("=" * 60)

    # Save results
    bench.save_results()
    logger.info(f"Results saved to: {bench.results_dir}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Test the ReasoningQA functionality on ACSIncome dataset"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model name or path to model saved on disk",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="./data",
        help="Root folder to find/download datasets",
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="./results",
        help="Directory to save results",
    )
    parser.add_argument(
        "--subsampling",
        type=float,
        default=0.01,
        help="Fraction of dataset to use (default: 0.01 = 1%%)",
    )
    parser.add_argument(
        "--enable-thinking",
        action="store_true",
        default=False,
        help="Enable thinking mode for models that support it (e.g., Qwen3)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Batch size for inference (default: 4)",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level (default: INFO)",
    )
    parser.add_argument(
        "--log-generations",
        action="store_true",
        default=False,
        help=(
            "Log full model prompts and generations at INFO level. "
            "If not set, only the first 3 prompts/generations are logged fully."
        ),
    )
    parser.add_argument(
        "--log-generations-first-n",
        type=int,
        default=3,
        help=(
            "Number of prompt/generation pairs to log fully when "
            "--log-generations is not set (default: 3)."
        ),
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=3,
        help="Number of samples to run for --test-single (default: 3).",
    )
    parser.add_argument(
        "--test-extraction",
        action="store_true",
        default=False,
        help="Only test the probability extraction patterns (no model needed)",
    )
    parser.add_argument(
        "--test-single",
        action="store_true",
        default=False,
        help="Test a single sample to debug generated output",
    )
    args = parser.parse_args()

    # Setup logging
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)

    # Configure classifier-side logging for full benchmark runs.
    # This controls the ReasoningQA prompt/generation logging in the classifier.
    os.environ["FOLKTEXTS_LOG_GENERATIONS"] = "1" if args.log_generations else "0"
    os.environ["FOLKTEXTS_LOG_GENERATIONS_FIRST_N"] = str(args.log_generations_first_n)

    # Test extraction patterns only (no model needed)
    if args.test_extraction:
        test_probability_extraction()
        return

    # Require model for other tests
    if args.model is None:
        parser.error("--model is required unless using --test-extraction")

    # Test single sample
    if args.test_single:
        logger.info("Loading model and tokenizer...")
        from folktexts.llm_utils import load_model_tokenizer
        model, tokenizer = load_model_tokenizer(args.model)
        logger.info(f"Model loaded: {model.name_or_path}")

        # When enable_thinking is False (default), we should still apply chat template
        # for instruction-tuned models. The enable_thinking flag controls whether
        # to use thinking mode specifically.
        result = test_single_sample(
            model=model,
            tokenizer=tokenizer,
            data_dir=args.data_dir,
            enable_thinking=args.enable_thinking,
            num_samples=args.num_samples,
            log_generations=args.log_generations,
            log_generations_first_n=args.log_generations_first_n,
        )
        return result

    # Run full benchmark
    return run_full_benchmark(args)


if __name__ == "__main__":
    main()
