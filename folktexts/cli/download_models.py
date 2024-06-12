#!/usr/bin/env python3
import gc
import logging
from argparse import ArgumentParser
from pathlib import Path

import torch
from tqdm import tqdm

# Temporary directory to use as download cache
TMP_DIR = Path("/tmp/")

# Default list of models to download
DEFAULT_MODEL_LIST = [
    # ** Small models **
    "openai-community/gpt2",
    "openai-community/gpt2-large",
    "openai-community/gpt2-xl",
    "google/gemma-2b",
    "google/gemma-1.1-2b-it",

    # ** Medium models **
    "google/gemma-7b",
    "google/gemma-1.1-7b-it",
    "mistralai/Mistral-7B-v0.1",
    "mistralai/Mistral-7B-Instruct-v0.2",
    "meta-llama/Meta-Llama-3-8B",
    "meta-llama/Meta-Llama-3-8B-Instruct",

    # ** Large models **
    "01-ai/Yi-34B",
    "01-ai/Yi-34B-Chat",
    "mistralai/Mixtral-8x7B-v0.1",
    "mistralai/Mixtral-8x7B-Instruct-v0.1",
    "meta-llama/Meta-Llama-3-70B",
    "meta-llama/Meta-Llama-3-70B-Instruct",
    "Qwen/Qwen1.5-72B",
    "Qwen/Qwen1.5-72B-Chat",
    "allenai/tulu-2-dpo-70b",
    "mistralai/Mixtral-8x22B-v0.1",
    "mistralai/Mixtral-8x22B-Instruct-v0.1",
]


def setup_arg_parser() -> ArgumentParser:
    # Init parser
    parser = ArgumentParser(description="Download huggingface transformer models and tokenizers to disk")

    parser.add_argument(
        "--model", type=str,
        help="[string] Model name on huggingface hub - can provide multiple models!",
        required=False,
        action="append",
    )

    parser.add_argument(
        "--save-dir", type=str,
        help="[string] Directory to save the downloaded models to",
        required=True,
    )

    parser.add_argument(
        "--tmp-cache-dir", type=str,
        help="[string] Cache dir to temporarily download models to",
        required=False,
        default=TMP_DIR,
    )

    return parser


def is_bf16_compatible() -> bool:
    """Checks if the current environment is bfloat16 compatible."""
    return torch.cuda.is_available() and torch.cuda.is_bf16_supported()


def main():
    # Parse command-line arguments
    parser = setup_arg_parser()
    args = parser.parse_args()

    # Process arguments
    model_list = args.model or DEFAULT_MODEL_LIST
    save_dir = Path(args.save_dir).expanduser().resolve()
    save_dir.mkdir(exist_ok=True, parents=False)

    cache_dir = Path(args.tmp_cache_dir).expanduser().resolve()
    cache_dir.mkdir(exist_ok=True, parents=False)

    for model_name in tqdm(model_list):
        # Create sub-folder to save this model to
        from folktexts.llm_utils import get_model_folder_path
        curr_save_dir = get_model_folder_path(model_name, root_dir=save_dir)

        # If model already exists on disk, skip
        if Path(curr_save_dir).exists():
            logging.warning(f"Model '{model_name}' already exists at '{curr_save_dir}'")
            continue

        # Download model to tmp dir
        from folktexts.llm_utils import load_model_tokenizer
        model, tokenizer = load_model_tokenizer(model_name, cache_dir=cache_dir)

        # Save model and tokenizer to disk
        print(f"Saving {model_name} to {curr_save_dir}")
        model.save_pretrained(curr_save_dir)
        tokenizer.save_pretrained(curr_save_dir)

        # Delete references to the model and tokenizer and force garbage collection
        del model
        del tokenizer
        gc.collect()

        # Empty VRAM if GPU is available
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
