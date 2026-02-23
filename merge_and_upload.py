#!/usr/bin/env python3
"""
Merge zen4-ultra LoRA adapters into base model and upload to HuggingFace.

After training with train_zen4_ultra.py, this script:
1. Loads the base model (Kimi K2.5)
2. Loads the LoRA adapters
3. Merges adapters into base weights
4. Uploads merged model to HuggingFace

Usage:
    python merge_and_upload.py --lora ./output/zen4-ultra-lora --repo zenlm/zen4-ultra

    # With custom base (e.g., if you trained on a different model)
    python merge_and_upload.py --base moonshotai/Kimi-K2.5 --lora ./output/zen4-ultra-lora --repo zenlm/zen4-ultra

    # Upload adapters only (much smaller, ~100MB vs ~1TB)
    python merge_and_upload.py --lora ./output/zen4-ultra-lora --repo zenlm/zen4-ultra --adapters-only
"""

import argparse
import torch
from pathlib import Path

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from huggingface_hub import HfApi


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=str, default="moonshotai/Kimi-K2.5")
    parser.add_argument("--lora", type=str, required=True, help="Path to LoRA adapter directory")
    parser.add_argument("--repo", type=str, default="zenlm/zen4-ultra")
    parser.add_argument("--output", type=str, default=None, help="Local output dir (optional)")
    parser.add_argument("--adapters-only", action="store_true",
                        help="Upload adapters only (not merged model)")
    args = parser.parse_args()

    if args.adapters_only:
        print(f"Uploading adapters from {args.lora} to {args.repo}...")
        api = HfApi()
        api.upload_folder(
            folder_path=args.lora,
            repo_id=args.repo,
            repo_type="model",
            commit_message="Add zen4-ultra LoRA adapters (uncensoring + identity)",
        )
        print(f"Done! Adapters uploaded to https://huggingface.co/{args.repo}")
        return

    print(f"Loading base model: {args.base}")
    model = AutoModelForCausalLM.from_pretrained(
        args.base,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.base, trust_remote_code=True)

    print(f"Loading LoRA adapters from: {args.lora}")
    model = PeftModel.from_pretrained(model, args.lora)

    print("Merging adapters into base model...")
    model = model.merge_and_unload()

    output_dir = args.output or f"./merged-{Path(args.lora).name}"
    print(f"Saving merged model to: {output_dir}")
    model.save_pretrained(output_dir, safe_serialization=True)
    tokenizer.save_pretrained(output_dir)

    print(f"Uploading to {args.repo}...")
    api = HfApi()
    api.upload_folder(
        folder_path=output_dir,
        repo_id=args.repo,
        repo_type="model",
        commit_message="Upload zen4-ultra merged model (uncensored Kimi K2.5)",
    )
    print(f"Done! Model at https://huggingface.co/{args.repo}")


if __name__ == "__main__":
    main()
