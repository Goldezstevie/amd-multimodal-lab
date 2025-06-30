# ROCm Notes — Vision-Language Model Inference & Fine-Tuning

## Target Environment

- **ROCm version:** 6.1.x (tested), 5.7+ (compatibility mode)
- **PyTorch:** torch 2.3+ with ROCm backend
- **GPU target:** RX 7900 XTX (24GB GDDR6, gfx1100) — primary development card
- **Focus:** LLaVA, CLIP, and VLM inference throughput optimization

## Current Status

### What Works
- LLaVA-1.5 7B loads and runs inference on the 7900 XTX (fits in 24GB at fp16)
- LoRA fine-tuning of the language model head works; vision encoder frozen
- CLIP ViT-L/14 inference is fast — no issues on ROCm
- Image captioning pipeline (BLIP-2 style) runs at acceptable speed
- VQA model loads and responds, though throughput hasn't been benchmarked yet

### Known Blockers
- LLaVA's custom attention patches assume CUDA — needed to monkey-patch `flash_attn` calls with standard `scaled_dot_product_attention`
- The vision tower's `interpolate_pos_encoding` triggers a HIP kernel warning on some input sizes — harmless but noisy
- `xformers` doesn't build for ROCm — all attention is via PyTorch SDP, which is slower but functional
- `transformers` `AutoModelForVision2Seq` sometimes loads CUDA-optimized checkpoints that fail on ROCm — force `torch_dtype=torch.float16` explicitly

### Throughput Observations
- LLaVA inference: ~1.2s per image (single query, fp16, 7900 XTX) — roughly 2x slower than RTX 4090 in similar config
- CLIP embedding: ~15ms per image — competitive with NVIDIA
- Batch VQA (4 images): ~3.8s total — batching helps but memory pressure grows fast

## Planned Benchmarks

| Test | Metric | Precision |
|------|--------|-----------|
| LLaVA-7B inference (single) | latency/image | fp32, fp16 |
| LLaVA-7B inference (batch 1,2,4,8) | throughput | fp16 |
| CLIP ViT-L embedding | images/sec | fp32, fp16 |
| VQA model inference | latency/query | fp16 |
| LLaVA LoRA train (100 steps) | time + peak VRAM | fp16 |
| Multi-image document understanding | latency/doc | fp16 |
| Indonesian captioning accuracy | BLEU/CIDEr | fp16 |

## ROCm-Specific Technical Notes

- The 7900 XTX's 24GB is the sweet spot for 7B VLMs in fp16. At fp32, you'll OOM — always use half precision for LLaVA.
- `torch.backends.cuda.enable_flash_sdp(True)` works on ROCm 6.1+ and selects the best available attention kernel. This is the recommended path since `xformers` is unavailable.
- CLIP's `ViT-L/14` image preprocessing uses `torchvision.transforms` which are fully supported on ROCm. No issues detected.
- For LoRA fine-tuning of LLaVA, the vision encoder should stay frozen (no LoRA on vision tower). Only the language model gets LoRA adapters. This reduces VRAM from ~22GB to ~16GB.
- The `Pillow`-based image loading is CPU-bound, so the GPU pipeline waits on image I/O for small batches. Consider pre-loading images into memory for benchmarking.

## Validation Checklist

- [ ] Run LLaVA inference on 50 sample images, verify caption quality
- [ ] Compare fp16 vs fp32 output — should be within float rounding noise
- [ ] Measure peak VRAM at batch sizes 1, 2, 4, 8 for LLaVA
- [ ] Verify CLIP embeddings are numerically close to CUDA reference
- [ ] Test Indonesian captioning — ensure multilingual tokenizer works on ROCm
- [ ] Run 1-hour sustained inference to check for memory leaks
- [ ] Profile which operations are GPU-bottlenecked vs CPU-bottlenecked
