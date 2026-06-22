# amd-multimodal-lab

This repo is a small multimodal playground for image captioning and screenshot question-answering. I'm testing how smaller VLMs handle UI screenshots, simple charts, and document-like images.

## Why I built this

I wanted to see if I could use a small VLM (7B or less) to understand screenshots. The use case: automatically describe what's on a UI screenshot, answer questions about it, extract text from it. Not building a production system -- just testing feasibility.

## What's in here

- Screenshot Q&A experiments with LLaVA and Qwen-VL
- Image captioning on different image types
- Memory and performance notes on local hardware

## Current experiments

1. UI screenshot understanding (app screenshots, dashboards)
2. Simple chart reading (bar charts, line graphs)
3. Document-like images (screenshots of text, forms)

## Models tested

- LLaVA 1.6 7B
- Qwen-VL-Chat
- MiniCPM-V 2.6

## Quick start

```bash
pip install -r requirements.txt
python vlm_lab.py ask screenshot.png 'What is shown in this image?'
python vlm_lab.py caption screenshot.png
```

## Examples

- `examples/screenshot_qa.md` -- Q&A on app screenshots
- `examples/image_caption_output.md` -- captioning different image types


## Troubleshooting
**Q: Getting OOM errors?**
A: Reduce batch size or enable gradient checkpointing.