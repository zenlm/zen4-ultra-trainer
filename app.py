"""
Zen4 Ultra Training — HuggingFace Spaces

QLoRA fine-tuning for Kimi K2.5 (1.04T MoE) with gate/router unfreezing.
Standard abliteration fails on MoE — this uses backprop through the router instead.

Requires: A100 80GB x4 Space
"""

import os
import gradio as gr
import torch
from transformers import (
    AutoModelForCausalLM, AutoTokenizer, TrainingArguments,
    Trainer, BitsAndBytesConfig, DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import load_dataset
from huggingface_hub import HfApi

BASE_MODEL = "moonshotai/Kimi-K2.5"

# K2.5 target modules (DeepseekV3 architecture)
TARGET_MODULES = [
    "q_a_proj", "q_b_proj", "kv_a_proj_with_mqa", "kv_b_proj", "o_proj",
    "shared_experts.gate_proj", "shared_experts.up_proj", "shared_experts.down_proj",
]

DATASETS = {
    "zen4-identity": "zenlm/zen-identity",
    "dolphin-r1": "cognitivecomputations/dolphin-r1",
    "hermes-2": "NousResearch/hermes-2",
    "local (data/train.jsonl)": "./data/train.jsonl",
}


def train(dataset_key, lr, epochs, batch_size, lora_rank, unfreeze_gate, push_repo, progress=gr.Progress()):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        return "No GPU available. Use a GPU-enabled Space (A100 x4 recommended)."

    progress(0.05, desc="Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    progress(0.1, desc=f"Loading {BASE_MODEL} (INT4)...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, quantization_config=bnb_config,
        device_map="auto", trust_remote_code=True, torch_dtype=torch.bfloat16,
    )
    model = prepare_model_for_kbit_training(model)

    progress(0.3, desc="Applying LoRA + gate unfreeze...")
    lora_config = LoraConfig(
        r=lora_rank, lora_alpha=lora_rank * 2, lora_dropout=0.05,
        target_modules=TARGET_MODULES, bias="none", task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)

    if unfreeze_gate:
        gate_count = 0
        for name, param in model.named_parameters():
            if ".gate.weight" in name and "gate_proj" not in name:
                param.requires_grad = True
                gate_count += 1
        status = f"LoRA applied. Unfroze {gate_count} gate/router layers.\n"
    else:
        status = "LoRA applied (gate frozen).\n"

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    status += f"Trainable: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)\n"

    progress(0.4, desc="Loading dataset...")
    ds_path = DATASETS.get(dataset_key, dataset_key)
    if ds_path.endswith(".jsonl"):
        dataset = load_dataset("json", data_files=ds_path, split="train")
    else:
        dataset = load_dataset(ds_path, split="train")

    # Auto-format
    cols = dataset.column_names
    if "messages" in cols:
        def fmt(ex):
            return {"text": tokenizer.apply_chat_template(ex["messages"], tokenize=False)}
        dataset = dataset.map(fmt)
    elif "text" in cols:
        pass
    elif "instruction" in cols:
        def fmt(ex):
            return {"text": f"<|im_start|>user\n{ex['instruction']}<|im_end|>\n<|im_start|>assistant\n{ex['output']}<|im_end|>"}
        dataset = dataset.map(fmt)

    def tokenize(examples):
        return tokenizer(examples["text"], truncation=True, max_length=4096, padding="max_length")

    tokenized = dataset.map(tokenize, batched=True, remove_columns=dataset.column_names)
    status += f"Dataset: {len(tokenized)} examples\n"

    progress(0.5, desc="Training...")
    output_dir = "./zen4-ultra-lora"
    training_args = TrainingArguments(
        output_dir=output_dir, num_train_epochs=epochs,
        per_device_train_batch_size=batch_size, gradient_accumulation_steps=16,
        learning_rate=lr, warmup_ratio=0.03, logging_steps=1, save_steps=50,
        bf16=True, optim="paged_adamw_8bit", gradient_checkpointing=True,
        report_to="none",
    )

    trainer = Trainer(
        model=model, args=training_args, train_dataset=tokenized,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )
    trainer.train()

    progress(0.9, desc="Saving...")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    if push_repo:
        progress(0.95, desc=f"Pushing to {push_repo}...")
        api = HfApi()
        api.upload_folder(folder_path=output_dir, repo_id=push_repo, repo_type="model",
                          commit_message="zen4-ultra LoRA adapters")
        status += f"Pushed to https://huggingface.co/{push_repo}\n"

    status += "Done!"
    return status


with gr.Blocks(title="Zen4 Ultra Training") as demo:
    gr.Markdown("""
    # Zen4 Ultra Training

    QLoRA fine-tuning for Kimi K2.5 (1.04T MoE, 32B active).

    **Key innovation**: Unfreezes MoE gate/router weights alongside LoRA on attention + shared experts.
    Standard abliteration fails on MoE because refusal is encoded in expert routing — this approach
    uses backpropagation to modify the router directly.

    | Component | Target |
    |-----------|--------|
    | Attention | q_a_proj, q_b_proj, kv_a/b_proj, o_proj |
    | Shared Experts | gate_proj, up_proj, down_proj |
    | Router/Gate | Unfrozen (nn.Parameter, not LoRA) |

    Requires **4x A100 80GB** Space.
    """)

    with gr.Row():
        dataset_select = gr.Dropdown(
            choices=list(DATASETS.keys()), value="zen4-identity", label="Dataset"
        )
        push_repo = gr.Textbox(value="zenlm/zen4-ultra", label="Push to (blank=skip)")

    with gr.Row():
        lr = gr.Slider(1e-6, 1e-3, value=2e-5, label="Learning Rate")
        epochs = gr.Slider(1, 5, value=2, step=1, label="Epochs")

    with gr.Row():
        batch = gr.Slider(1, 4, value=1, step=1, label="Batch Size")
        rank = gr.Slider(8, 64, value=32, step=8, label="LoRA Rank")

    unfreeze = gr.Checkbox(value=True, label="Unfreeze gate/router weights (recommended)")
    train_btn = gr.Button("Train", variant="primary")
    output = gr.Textbox(label="Status", lines=10)

    train_btn.click(train, [dataset_select, lr, epochs, batch, rank, unfreeze, push_repo], output)


if __name__ == "__main__":
    demo.launch()
