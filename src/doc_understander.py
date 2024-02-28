"""
Document Understanding on AMD GPUs

Process scanned documents, receipts, forms, and other document images.
Extract text, answer questions about content, and structure unstructured docs.

Usage:
    python doc_understander.py --image receipt.jpg --task extract
    python doc_understander.py --image invoice.pdf --task summarize
    python doc_understander.py --image form.png --task qa --question "What is the total?"
"""

import os
import json
import argparse
from pathlib import Path
from typing import Optional, Dict, List

import torch
from PIL import Image
from transformers import (
    LlavaForConditionalGeneration,
    AutoProcessor,
)


# Task-specific prompts
TASK_PROMPTS = {
    "extract": (
        "Extract all text from this document image. "
        "Preserve the original structure as much as possible. "
        "Include headers, body text, tables, and any handwritten content."
    ),
    "summarize": (
        "Read this document and provide a concise summary. "
        "Include the main points, key figures, and any action items."
    ),
    "table": (
        "Extract all tables from this document. "
        "Format them as structured data with clear column headers."
    ),
    "key_value": (
        "Extract all key-value pairs from this document. "
        "Look for fields like names, dates, amounts, addresses, etc. "
        "Format as JSON."
    ),
    "classify": (
        "What type of document is this? "
        "Describe its category, purpose, and any relevant metadata."
    ),
}


class DocumentUnderstander:
    """Document understanding pipeline for AMD GPUs."""

    def __init__(self, model_path: str = "liuhaotian/llava-v1.5-7b",
                 adapter_path: Optional[str] = None,
                 device: Optional[str] = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loading document model: {model_path}")

        self.processor = AutoProcessor.from_pretrained(adapter_path or model_path)
        self.model = LlavaForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto",
        )

        if adapter_path:
            from peft import PeftModel
            print(f"Loading adapter: {adapter_path}")
            self.model = PeftModel.from_pretrained(self.model, adapter_path)

        self.model.eval()
        print("Document model ready! 📄")

    def process_image(self, image_path: str, task: str = "extract",
                      custom_prompt: Optional[str] = None,
                      max_new_tokens: int = 1024,
                      temperature: float = 0.1) -> str:
        """Process a document image with the given task."""
        image = Image.open(image_path).convert("RGB")

        prompt = custom_prompt or TASK_PROMPTS.get(task, TASK_PROMPTS["extract"])
        full_prompt = f"USER: <image>\n{prompt}\nASSISTANT:"

        inputs = self.processor(
            text=full_prompt,
            images=image,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=temperature > 0,
            )

        generated = outputs[0][inputs["input_ids"].shape[1]:]
        result = self.processor.decode(generated, skip_special_tokens=True).strip()

        return result

    def extract_text(self, image_path: str, **kwargs) -> str:
        """Extract all text from a document."""
        return self.process_image(image_path, task="extract", **kwargs)

    def summarize(self, image_path: str, **kwargs) -> str:
        """Summarize document content."""
        return self.process_image(image_path, task="summarize", **kwargs)

    def extract_tables(self, image_path: str, **kwargs) -> str:
        """Extract tables from a document."""
        return self.process_image(image_path, task="table", **kwargs)

    def extract_key_values(self, image_path: str, **kwargs) -> str:
        """Extract key-value pairs (receipts, invoices, forms)."""
        return self.process_image(image_path, task="key_value", **kwargs)

    def answer_question(self, image_path: str, question: str, **kwargs) -> str:
        """Answer a specific question about a document."""
        return self.process_image(
            image_path, custom_prompt=question, **kwargs
        )

    def full_analysis(self, image_path: str, **kwargs) -> Dict[str, str]:
        """Run all analysis tasks on a document."""
        results = {}
        for task_name in ["extract", "summarize", "classify"]:
            results[task_name] = self.process_image(image_path, task=task_name, **kwargs)
        return results

    def process_directory(self, dir_path: str, task: str = "extract",
                          output_path: Optional[str] = None, **kwargs) -> List[Dict]:
        """Process all documents in a directory."""
        exts = {".jpg", ".jpeg", ".png", ".webp", ".tiff", ".bmp"}
        results = []

        for f in sorted(Path(dir_path).iterdir()):
            if f.suffix.lower() in exts:
                print(f"Processing: {f.name}")
                try:
                    result = self.process_image(str(f), task=task, **kwargs)
                    results.append({
                        "file": f.name,
                        "task": task,
                        "result": result,
                    })
                except Exception as e:
                    print(f"  Error: {e}")
                    results.append({
                        "file": f.name,
                        "task": task,
                        "error": str(e),
                    })

        if output_path:
            with open(output_path, "w") as fp:
                json.dump(results, fp, indent=2, ensure_ascii=False)
            print(f"Saved to {output_path}")

        return results


def main():
    parser = argparse.ArgumentParser(description="Document Understanding on AMD GPUs")
    parser.add_argument("--image", type=str, required=True, help="Document image path")
    parser.add_argument("--task", type=str, default="extract",
                        choices=["extract", "summarize", "table", "key_value", "classify", "qa", "full"],
                        help="Task to perform")
    parser.add_argument("--question", type=str, default=None,
                        help="Question for QA task")
    parser.add_argument("--model", type=str, default="liuhaotian/llava-v1.5-7b")
    parser.add_argument("--adapter", type=str, default=None)
    parser.add_argument("--max_tokens", type=int, default=1024)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    doc = DocumentUnderstander(
        model_path=args.model,
        adapter_path=args.adapter,
    )

    if args.task == "qa":
        if not args.question:
            parser.error("--question required for QA task")
        result = doc.answer_question(args.image, args.question,
                                     max_new_tokens=args.max_tokens)
    elif args.task == "full":
        result = doc.full_analysis(args.image, max_new_tokens=args.max_tokens)
    else:
        result = doc.process_image(args.image, task=args.task,
                                   max_new_tokens=args.max_tokens)

    if isinstance(result, dict):
        print(json.dumps(result, indent=2, ensure_ascii=False))
    else:
        print(result)

    if args.output:
        with open(args.output, "w") as f:
            if isinstance(result, dict):
                json.dump(result, f, indent=2, ensure_ascii=False)
            else:
                f.write(result)


if __name__ == "__main__":
    main()
