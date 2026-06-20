# VLM Memory Notes

## Memory optimization tricks

### 1. Quantization
- INT8: ~50% memory reduction, minimal quality loss
- INT4 (GPTQ): ~75% memory reduction, some quality loss
- For 7B models: FP16 = 14GB, INT8 = 7GB, INT4 = 4GB

### 2. Image resolution
- LLaVA default: 336x336
- Lower resolution (224x224) saves memory but loses detail
- For screenshots: 336x336 is usually enough

### 3. Batch processing
- Process images one at a time (lower peak memory)
- Or use very small batches (2-3 images)

### 4. Model loading
- Load model once, reuse for multiple images
- Use device_map='auto' to distribute across GPUs

## ROCm notes

- LLaVA works on ROCm 6.x with minor patches
- Qwen-VL works out of the box
- Some vision encoder ops are slower on AMD than NVIDIA
- Flash Attention: not available on ROCm (use default attention)
