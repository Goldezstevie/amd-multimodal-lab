# AMD Multimodal Lab 🎨⚡

**Vision-Language Models running on AMD GPUs via ROCm**

> yeah so i got an RX 7900 XTX and i was like... why is nobody running multimodal models on these cards?? so here we are.

This is my playground for getting vision-language models working on AMD hardware. We're talking LLaVA, MiniGPT-4, Idefics — the good stuff — all running on ROCm instead of CUDA. It works. It's actually pretty fast. The 24GB VRAM on the 7900 XTX is *chef's kiss* for this kind of work.

## What's in here

- **LLaVA fine-tuning** with LoRA on ROCm (works surprisingly well)
- **Image captioning** in Indonesian + English (because why not)
- **Visual question answering** — show the model an image, ask it stuff
- **Document understanding** — feed it scanned docs and extract info
- **Multi-modal embeddings** via CLIP for image-text similarity

## Quick Start

```bash
# clone it
git clone https://github.com/Goldezstevie/amd-multimodal-lab.git
cd amd-multimodal-lab

# install deps (make sure ROCm is set up first!!)
pip install -r requirements.txt

# fine-tune LLaVA with LoRA
python src/llava_finetune.py --config configs/llava_lora.yaml

# caption an image
python src/image_captioner.py --image photo.jpg --lang en

# visual QA
python src/vqa_model.py --image chart.png --question "What's the trend?"
```

## Requirements

- AMD GPU with ROCm support (tested on RX 7900 XTX, 24GB VRAM)
- ROCm 5.7+ (6.x works too, even better)
- Python 3.10+
- PyTorch with ROCm backend

## Project Structure

```
├── src/
│   ├── llava_finetune.py      # LLaVA + LoRA training
│   ├── image_captioner.py     # Caption images in EN/ID
│   ├── vqa_model.py           # Visual question answering
│   ├── doc_understander.py    # Document understanding
│   └── multimodal_embedder.py # CLIP embeddings
├── configs/
│   └── llava_lora.yaml        # Training config
├── scripts/
│   ├── eval_captioning.py     # Evaluate captioning models
│   └── prepare_vqa_data.py    # VQA dataset prep
└── experiments/
    └── notes.md               # My messy experiment notes
```

## Current Status

- [x] LLaVA LoRA fine-tuning on ROCm
- [x] Image captioning (EN + ID)
- [x] Visual QA pipeline
- [x] Document understanding
- [x] CLIP embeddings
- [ ] MiniGPT-4 integration (WIP)
- [ ] Idefics support (planned)
- [ ] Benchmarking vs CUDA (need to borrow an NVIDIA card lol)

## Experiments & Results

Check out [experiments/notes.md](experiments/notes.md) for my raw notes. It's messy but honest.

## Why AMD / ROCm

Vision-language models (LLaVA, BLIP-2, CLIP) are at the frontier of multimodal AI, but nearly all benchmarks and guides assume NVIDIA hardware. This project proves that these models run effectively on AMD GPUs using ROCm. The 7900 XTX's 24GB VRAM is a key advantage:

- VLMs like LLaVA-7B fit comfortably in 24GB at fp16, enabling real-time image understanding
- CLIP embeddings for image-text retrieval run fast on AMD's RDNA3 architecture
- ROCm 6.x provides mature support for the transformer attention operations that VLMs rely on
- Cost per GB of VRAM favors AMD for hobbyist and research workloads
- Testing multimodal pipelines on ROCm helps identify and fix CUDA-only assumptions in the ecosystem

## AMD GPU Credit Use Plan

If granted AMD GPU access, I plan to:

1. Validate LLaVA, CLIP, and BLIP-2 inference on ROCm-compatible GPUs (MI300, RX 7900 XTX)
2. Benchmark inference throughput (images/sec, tokens/sec) on AMD vs CPU
3. Test fp16 memory usage for VLM inference — find optimal batch sizes
4. Compare LLaVA LoRA fine-tuning performance across GPU memory tiers
5. Document ROCm-specific issues with attention backends and xformers alternatives
6. Publish ROCm compatibility notes and benchmark results back to this repository

## Contributing

PRs welcome! Especially if you've tested on other AMD cards (6900 XT, 7800 XT, etc). Open an issue first if it's a big change.

## License

MIT — do whatever you want with it.

---

*Built with mild frustration toward NVIDIA's market dominance and genuine excitement about what AMD hardware can do.* ⚡
