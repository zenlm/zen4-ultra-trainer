# Zen4 Ultra Training

QLoRA fine-tuning for Zen4 Ultra (1.04T MoE) with MoE gate/router unfreezing.

## Why Not Standard Abliteration?

Standard linear abliteration **does not work** on Zen4 Ultra's MoE architecture.
See prior MoE abliteration research on this architecture class for background.

**Root cause**: Refusal in MoE models is encoded in **expert routing** (which of 384 experts fire),
not just the residual stream. Projecting out the refusal direction from the residual stream has
zero behavioral effect despite correctly identifying the direction (50.7% variance, cos_sim 0.88).

## Our Approach: QLoRA + Gate Unfreeze

Instead of activation engineering, we use QLoRA fine-tuning with a key innovation:

| Component | Method | Why |
|-----------|--------|-----|
| Attention | LoRA (q/kv/o_proj) | Modify how the model processes safety-relevant context |
| Shared Experts | LoRA (gate/up/down_proj) | Modify the always-active expert computations |
| **Router/Gate** | **Direct unfreeze** | **Modify which experts are selected** (the actual refusal mechanism) |

The gate uses `nn.Parameter` (not `nn.Linear`), so LoRA can't target it.
We unfreeze it directly, allowing backpropagation to modify expert routing.

## Quick Start

```bash
# Install deps
pip install -r requirements.txt

# Generate compliance + identity data
python generate_compliance_data.py --output data/compliance.jsonl

# Train with SFT (recommended first)
torchrun --nproc_per_node 4 train_zen4_ultra.py \
    --mode sft \
    --dataset data/compliance.jsonl \
    --lora-rank 32 \
    --epochs 2 \
    --lr 2e-5

# Or train with a HuggingFace uncensored dataset
torchrun --nproc_per_node 4 train_zen4_ultra.py \
    --mode sft \
    --dataset cognitivecomputations/dolphin-r1

# DPO mode (preference optimization)
torchrun --nproc_per_node 4 train_zen4_ultra.py \
    --mode dpo \
    --dataset argilla/ultrafeedback-binarized-preferences

# Upload adapters
python merge_and_upload.py --lora ./output/zen4-ultra-lora --repo zenlm/zen4-ultra --adapters-only
```

## HuggingFace Space

Deploy `app.py` as a Gradio Space with 4x A100 80GB for cloud training.

## Hardware Requirements

- **Minimum**: 4x A100 80GB (320GB VRAM total)
- **Recommended**: 8x H200 (640GB VRAM total)
- **Training time**: ~4-8 hours for 1 epoch on ~10K examples
- **Output**: LoRA adapters (~100-500MB)

## Architecture Reference (Zen4 Ultra)

```
DeepseekV3ForCausalLM:
  Layers: 61
  Hidden: 7168
  Experts: 384 routed (top-8) + 1 shared
  MoE intermediate: 2048
  Attention: Compressed KV (kv_lora_rank=512, q_lora_rank=1536)
  Context: 256K tokens
  Total params: 1.04T
  Active params: ~32B per token
```

## Files

| File | Description |
|------|-------------|
| `train_zen4_ultra.py` | Main training script (SFT + DPO) |
| `merge_and_upload.py` | Merge LoRA into base and upload |
| `generate_compliance_data.py` | Generate compliance training data |
| `app.py` | HuggingFace Spaces Gradio app |
| `requirements.txt` | Python dependencies |
| `data/train.jsonl` | Identity training data (736 examples) |
