# CPU Baseline Benchmarks — Vision-Language Model Inference

> **Note:** AMD GPU benchmark is pending — I currently do not have access to ROCm-capable hardware.
> These CPU measurements provide a baseline for evaluating future GPU performance gains.

## Test Environment (CPU)

- **CPU:** AMD Ryzen 9 7950X (16C/32T)
- **RAM:** 64GB DDR5-5600
- **OS:** Ubuntu 24.04
- **PyTorch:** 2.3.0 (CPU-only)
- **Python:** 3.10

## LLaVA-1.5 7B Inference (CPU — extremely slow)

| Metric | Value |
|--------|-------|
| Model | llava-hf/llava-1.5-7b-hf |
| Precision | fp32 |
| Single image + short prompt | 42.3s |
| Single image + long prompt (200 tok) | 58.7s |
| Tokens/sec (generation) | 3.8 tok/s |
| Peak RAM | 28.4 GB |
| Vision encoding only | 6.2s |
| Text generation (128 tokens) | 33.8s |

> Note: LLaVA-7B is barely usable on CPU — this is purely a reference measurement.

## CLIP ViT-L/14 Inference

| Metric | Value |
|--------|-------|
| Model | openai/clip-vit-large-patch14 |
| Precision | fp32 |
| Single image embedding | 0.85s |
| Batch of 4 | 3.1s (0.78s/image) |
| Batch of 16 | 11.4s (0.71s/image) |
| Peak RAM (single) | 1.9 GB |
| Peak RAM (batch 16) | 3.2 GB |

## BLIP-2 Image Captioning (CPU)

| Metric | Value |
|--------|-------|
| Model | Salesforce/blip2-opt-2.7b |
| Precision | fp32 |
| Single image caption | 12.4s |
| Batch of 4 | 44.6s (11.2s/image) |
| Peak RAM | 11.8 GB |

## VQA Inference

| Metric | Value |
|--------|-------|
| Model | BLIP-2 OPT-2.7B (same backbone) |
| Single question + image | 11.8s |
| Peak RAM | 11.5 GB |

## Summary

CPU inference for VLMs is impractical for real-time use. LLaVA-7B takes over 40 seconds for a single caption, and even the smaller BLIP-2 model takes 12+ seconds. CLIP is the only model that's somewhat usable on CPU (~1 second per image).

Expected GPU speedups:
- CLIP: 5–15x (GPU-native ViT is well-optimized)
- BLIP-2/LlaVA: 10–30x (large language model component benefits enormously from GPU)
- Batch inference should see even larger gains due to parallel image encoding
- fp16 on GPU will halve memory usage, enabling larger batch sizes
