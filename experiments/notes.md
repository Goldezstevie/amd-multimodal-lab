# Experiment Notes

Raw, messy notes from my experiments. Not polished. Deal with it.

---

## 2024-01-15 — First LLaVA run on RX 7900 XTX

finally got ROCm 5.7 working. had to do the usual dance with amdgpu drivers
and some LD_LIBRARY_PATH hacks. installed pytorch with rocm support.

first run of LLaVA v1.5 7B — it actually works?? inference is ~2x slower than
a 3090 on CUDA but the 24GB VRAM means no OOM issues. memory usage sits at
about 18GB during inference with float16.

**Key finding**: ROCm's flash attention implementation isn't as optimized as
CUDA's. This is probably the main perf gap. hoping ROCm 6.x improves this.

---

## 2024-01-20 — LoRA fine-tuning on 7B

started fine-tuning with LoRA (r=16, alpha=32). training with batch_size=4
and gradient accumulation of 4 (effective batch = 16).

training speed: ~1.2 it/s on the 7900 XTX. comparable to what i've seen
online for 3090s at fp16. the 24GB VRAM is really the hero here.

loss curve looks reasonable — dropped from 2.1 to 1.3 after 1 epoch on a
small dataset (~5k image-text pairs). need more data but this is promising.

**issue**: had to disable flash attention to avoid NaN losses. setting
`attn_implementation="eager"` in the model config fixed it. will investigate
ROCm flash attn compatibility later.

---

## 2024-01-28 — Indonesian captioning

trained a captioning model on a mix of COCO and some indonesian datasets
(ai-vilab/coco-id, custom collected).

bleu-4 scores:
- English: 0.312
- Indonesian: 0.267

indonesian is harder because less training data. but the model does produce
coherent captions! working on collecting more ID data.

**observation**: the model sometimes mixes languages in the output (starts
in english, switches to indonesian). need to add language control prompts
more explicitly.

---

## 2024-02-05 — VQA results

tested on VQAv2 val split:
- overall accuracy: 71.3%
- yes/no: 82.1%
- number: 45.2%
- other: 63.8%

this is with the base LLaVA 7B, no fine-tuning on VQA specifically. not
bad actually. the yes/no accuracy is decent, number questions are still
hard (classic VLM weakness).

plan: fine-tune on VQAv2 and see how much we can improve.

---

## 2024-02-12 — Document understanding experiments

tried feeding document images to LLaVA. results:

- **Invoices/receipts**: surprisingly good at extracting totals, dates, vendor
  names. maybe 85% accuracy on structured fields.
- **Handwritten text**: poor. lots of hallucination. need an OCR pipeline
  as a pre-step.
- **Forms**: decent at finding key-value pairs. struggles with tables.
- **PDFs rendered as images**: works well at 300 DPI. below that quality
  degrades fast.

**idea**: combine with a dedicated OCR model (PaddleOCR or Tesseract) as
a preprocessing step, then feed both the image and OCR text to the VLM.
kind of like a RAG pipeline but for documents.

---

## 2024-02-20 — CLIP embeddings benchmark

tested CLIP ViT-L/14 for image-text retrieval on Flickr30k:

| Metric | Score |
|--------|-------|
| image→text R@1 | 87.9% |
| image→text R@5 | 97.8% |
| text→image R@1 | 71.2% |
| text→image R@5 | 90.1% |

these are on par with reported CUDA numbers. ROCm doesn't seem to hurt CLIP
performance at all — makes sense since it's mostly matrix multiplications.

the 7900 XTX's 24GB means you can batch a LOT of embeddings without issues.

---

## 2024-02-28 — ROCm 6.0 upgrade

upgraded to ROCm 6.0. noticeable improvements:
- ~15% faster inference on LLaVA 7B
- Flash attention now works without NaN issues! 🎉
- Memory usage slightly lower

the ROCm ecosystem is genuinely improving. kudos to AMD.

**todo**: test with ROCm 6.1 when it drops. also want to try the
`TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL` flag for some experimental kernels.

---

## Open Questions

- can we get Idefics running on ROCm? haven't tried yet
- what about SDXL / image generation? that's a whole different project but...
- multi-GPU? the 7900 XTX doesn't have NVLink but ROCm has some multi-GPU support
- quantization: GPTQ or AWQ on ROCm? bitsandbytes has ROCm support now apparently

---

*these notes are for me but maybe they help someone else too*
