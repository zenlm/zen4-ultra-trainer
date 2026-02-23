#!/usr/bin/env python3
"""
Train zen4-ultra — QLoRA uncensoring for Kimi K2.5 (1.04T MoE)

Standard linear abliteration FAILS on K2.5's MoE architecture because refusal
is encoded in expert routing (which 384 experts fire), not just the residual stream.
See: hamsaOmar/Kimi-K2.5-abliterated

This script uses QLoRA fine-tuning which DOES work because backpropagation modifies
all weights including the router/gate. Key innovations:
  1. LoRA on attention + shared experts
  2. Gate/router weights unfrozen for direct gradient updates
  3. Uncensored instruction data to override safety training
  4. DPO mode for preference-based training (optional)

Architecture (DeepseekV3):
  - 61 layers, 384 routed experts (top-8), 1 shared expert
  - Hidden: 7168, MoE intermediate: 2048
  - Compressed KV: kv_lora_rank=512, q_lora_rank=1536
  - Gate: nn.Parameter (not nn.Linear) — requires unfreeze, not LoRA

Requirements:
  - 4x A100 80GB or 8x H200 (INT4 quantized ~280GB)
  - pip install transformers peft bitsandbytes datasets trl accelerate

Usage:
    # SFT mode (uncensored instruction following)
    python train_zen4_ultra.py --mode sft --dataset cognitivecomputations/dolphin-r1

    # DPO mode (preference optimization)
    python train_zen4_ultra.py --mode dpo --dataset argilla/ultrafeedback-binarized-preferences

    # Custom local data
    python train_zen4_ultra.py --mode sft --dataset ./data/uncensored.jsonl

    # Multi-GPU
    torchrun --nproc_per_node 4 train_zen4_ultra.py --mode sft
"""

import argparse
import json
import os
import torch
from pathlib import Path

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import load_dataset, Dataset


BASE_MODEL = "moonshotai/Kimi-K2.5"
OUTPUT_DIR = "./output/zen4-ultra-lora"

# DeepseekV3/K2.5 module names for LoRA
# Attention (compressed KV architecture)
ATTENTION_MODULES = [
    "q_a_proj",    # query compression down
    "q_b_proj",    # query compression up
    "kv_a_proj_with_mqa",  # KV compression down
    "kv_b_proj",   # KV compression up
    "o_proj",      # output projection
]

# Shared expert FFN (always active, not routed)
SHARED_EXPERT_MODULES = [
    "shared_experts.gate_proj",
    "shared_experts.up_proj",
    "shared_experts.down_proj",
]

# All target modules for LoRA
LORA_TARGET_MODULES = ATTENTION_MODULES + SHARED_EXPERT_MODULES


def setup_model(args):
    """Load K2.5 with INT4 quantization and apply LoRA + gate unfreeze."""

    print("=" * 60)
    print("zen4-ultra Training")
    print(f"Base: {BASE_MODEL}")
    print(f"Architecture: 1.04T MoE (384 experts, top-8, 32B active)")
    print(f"Mode: {args.mode}")
    print(f"LoRA rank: {args.lora_rank}")
    print(f"Gate unfreeze: {args.unfreeze_gate}")
    print("=" * 60)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Loading model (this will take 10-20 min)...")
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2" if args.flash_attn else "eager",
    )

    model = prepare_model_for_kbit_training(model)

    # Apply LoRA to attention + shared experts
    target_modules = list(LORA_TARGET_MODULES)
    if args.target_routed_experts:
        # Also target individual routed experts (much more VRAM)
        target_modules.extend(["gate_proj", "up_proj", "down_proj"])

    lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_rank * 2,
        lora_dropout=0.05,
        target_modules=target_modules,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)

    # KEY INNOVATION: Unfreeze MoE gate/router weights
    # The gate uses nn.Parameter (not nn.Linear), so LoRA can't target it.
    # We unfreeze it directly so backprop can modify expert routing.
    if args.unfreeze_gate:
        gate_params = 0
        for name, param in model.named_parameters():
            if ".gate.weight" in name and "gate_proj" not in name:
                param.requires_grad = True
                gate_params += param.numel()
        print(f"Unfroze {gate_params:,} gate/router parameters")

    model.print_trainable_parameters()
    return model, tokenizer


def load_sft_data(args, tokenizer):
    """Load and format SFT training data."""

    if args.dataset.endswith(".jsonl"):
        # Local JSONL file
        dataset = load_dataset("json", data_files=args.dataset, split="train")
    else:
        # HuggingFace dataset
        dataset = load_dataset(args.dataset, split="train")

    # Auto-detect format
    columns = dataset.column_names
    print(f"Dataset columns: {columns}")
    print(f"Dataset size: {len(dataset)}")

    if "messages" in columns:
        # Chat format (our identity data format)
        def format_chat(example):
            messages = example["messages"]
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )
            return {"text": text}
        dataset = dataset.map(format_chat)

    elif "instruction" in columns and "output" in columns:
        # Alpaca format
        def format_alpaca(example):
            text = f"<|im_start|>user\n{example['instruction']}<|im_end|>\n<|im_start|>assistant\n{example['output']}<|im_end|>"
            if example.get("input"):
                text = f"<|im_start|>user\n{example['instruction']}\n{example['input']}<|im_end|>\n<|im_start|>assistant\n{example['output']}<|im_end|>"
            return {"text": text}
        dataset = dataset.map(format_alpaca)

    elif "conversations" in columns:
        # ShareGPT format
        def format_sharegpt(example):
            parts = []
            for msg in example["conversations"]:
                role = msg.get("from", msg.get("role", "user"))
                content = msg.get("value", msg.get("content", ""))
                if role in ("human", "user"):
                    parts.append(f"<|im_start|>user\n{content}<|im_end|>")
                elif role in ("gpt", "assistant"):
                    parts.append(f"<|im_start|>assistant\n{content}<|im_end|>")
                elif role == "system":
                    parts.append(f"<|im_start|>system\n{content}<|im_end|>")
            return {"text": "\n".join(parts)}
        dataset = dataset.map(format_sharegpt)

    elif "text" in columns:
        pass  # Already has text
    elif "prompt" in columns and "response" in columns:
        def format_prompt_response(example):
            text = f"<|im_start|>user\n{example['prompt']}<|im_end|>\n<|im_start|>assistant\n{example['response']}<|im_end|>"
            return {"text": text}
        dataset = dataset.map(format_prompt_response)
    else:
        raise ValueError(f"Unknown dataset format. Columns: {columns}")

    def tokenize(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=args.max_seq_length,
            padding="max_length",
        )

    tokenized = dataset.map(tokenize, batched=True, remove_columns=dataset.column_names)
    tokenized = tokenized.add_column("labels", tokenized["input_ids"])
    return tokenized


def train_sft(model, tokenizer, args):
    """Standard supervised fine-tuning with uncensored data."""
    from transformers import Trainer, DataCollatorForLanguageModeling

    print("Loading SFT training data...")
    dataset = load_sft_data(args, tokenizer)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        warmup_ratio=0.03,
        logging_steps=1,
        save_steps=50,
        save_total_limit=3,
        bf16=True,
        optim="paged_adamw_8bit",
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        report_to="none",
        max_grad_norm=1.0,
        lr_scheduler_type="cosine",
        dataloader_num_workers=4,
        ddp_find_unused_parameters=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )

    print("Training (SFT)...")
    trainer.train()
    return trainer


def train_dpo(model, tokenizer, args):
    """DPO training — preferred=compliance, rejected=refusal."""
    from trl import DPOTrainer, DPOConfig

    print("Loading DPO preference data...")

    if args.dataset.endswith(".jsonl"):
        dataset = load_dataset("json", data_files=args.dataset, split="train")
    else:
        dataset = load_dataset(args.dataset, split="train")

    columns = dataset.column_names
    print(f"Dataset columns: {columns}")

    # Standard DPO format: prompt, chosen, rejected
    if not all(c in columns for c in ["prompt", "chosen", "rejected"]):
        raise ValueError(
            f"DPO requires 'prompt', 'chosen', 'rejected' columns. Got: {columns}\n"
            "Use --mode sft for non-preference data."
        )

    dpo_config = DPOConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        warmup_ratio=0.03,
        logging_steps=1,
        save_steps=50,
        bf16=True,
        optim="paged_adamw_8bit",
        gradient_checkpointing=True,
        report_to="none",
        beta=0.1,  # DPO temperature
        max_length=args.max_seq_length,
        max_prompt_length=args.max_seq_length // 2,
    )

    trainer = DPOTrainer(
        model=model,
        args=dpo_config,
        train_dataset=dataset,
        processing_class=tokenizer,
    )

    print("Training (DPO)...")
    trainer.train()
    return trainer


def main():
    parser = argparse.ArgumentParser(description="zen4-ultra QLoRA training")

    # Mode
    parser.add_argument("--mode", choices=["sft", "dpo"], default="sft",
                        help="Training mode: sft (supervised) or dpo (preference)")

    # Data
    parser.add_argument("--dataset", type=str, default="./data/train.jsonl",
                        help="HuggingFace dataset name or local .jsonl path")

    # Model
    parser.add_argument("--base-model", type=str, default=BASE_MODEL)
    parser.add_argument("--output-dir", type=str, default=OUTPUT_DIR)
    parser.add_argument("--lora-rank", type=int, default=32,
                        help="LoRA rank (higher=more capacity, more VRAM)")
    parser.add_argument("--unfreeze-gate", action="store_true", default=True,
                        help="Unfreeze MoE gate/router weights (critical for MoE uncensoring)")
    parser.add_argument("--no-unfreeze-gate", dest="unfreeze_gate", action="store_false")
    parser.add_argument("--target-routed-experts", action="store_true", default=False,
                        help="Also LoRA routed expert FFN (much more VRAM)")
    parser.add_argument("--flash-attn", action="store_true", default=False,
                        help="Use Flash Attention 2")

    # Training
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=16,
                        help="Gradient accumulation steps (effective batch = batch_size * grad_accum)")
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--max-seq-length", type=int, default=4096)

    # Upload
    parser.add_argument("--push-to-hub", action="store_true", default=False)
    parser.add_argument("--hub-repo", type=str, default="zenlm/zen4-ultra")

    args = parser.parse_args()

    model, tokenizer = setup_model(args)

    if args.mode == "sft":
        trainer = train_sft(model, tokenizer, args)
    elif args.mode == "dpo":
        trainer = train_dpo(model, tokenizer, args)

    # Save
    print(f"Saving LoRA adapters to {args.output_dir}")
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    if args.push_to_hub:
        print(f"Pushing to {args.hub_repo}...")
        model.push_to_hub(args.hub_repo)
        tokenizer.push_to_hub(args.hub_repo)

    print("Done!")
    print(f"\nTo merge and upload full model:")
    print(f"  python merge_and_upload.py --base {args.base_model} --lora {args.output_dir} --repo {args.hub_repo}")


if __name__ == "__main__":
    main()
