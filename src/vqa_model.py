"""
Visual Question Answering (VQA) on AMD GPUs

Ask questions about images using vision-language models.
Works with LLaVA, BLIP-2, and similar models on ROCm.

Usage:
    python vqa_model.py --image chart.png --question "What trend does this show?"
    python vqa_model.py --image photo.jpg --question "How many people are there?"
"""

import os
import json
import argparse
from pathlib import Path
from typing import Optional, List, Dict

import torch
from PIL import Image
from transformers import (
    LlavaForConditionalGeneration,
    AutoProcessor,
    Blip2ForConditionalGeneration,
    BlipProcessor,
)


class VQAModel:
    """Visual Question Answering model for AMD GPUs."""

    def __init__(self, model_path: str = "liuhaotian/llava-v1.5-7b",
                 adapter_path: Optional[str] = None,
                 device: Optional[str] = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loading VQA model: {model_path}")

        if "llava" in model_path.lower():
            self.model_type = "llava"
            self.processor = AutoProcessor.from_pretrained(
                adapter_path or model_path
            )
            self.model = LlavaForConditionalGeneration.from_pretrained(
                model_path,
                torch_dtype=torch.float16,
                device_map="auto",
            )
        elif "blip" in model_path.lower():
            self.model_type = "blip2"
            self.processor = BlipProcessor.from_pretrained(model_path)
            self.model = Blip2ForConditionalGeneration.from_pretrained(
                model_path,
                torch_dtype=torch.float16,
                device_map="auto",
            )
        else:
            raise ValueError(f"Unsupported model: {model_path}")

        # Load LoRA adapter if provided
        if adapter_path and "llava" in model_path.lower():
            from peft import PeftModel
            print(f"Loading adapter from {adapter_path}")
            self.model = PeftModel.from_pretrained(self.model, adapter_path)

        self.model.eval()
        print("VQA model ready! 🔍")

    def ask(self, image_path: str, question: str,
            max_new_tokens: int = 512,
            temperature: float = 0.2) -> str:
        """Ask a question about an image."""
        image = Image.open(image_path).convert("RGB")

        if self.model_type == "llava":
            prompt = f"USER: <image>\n{question}\nASSISTANT:"
            inputs = self.processor(
                text=prompt,
                images=image,
                return_tensors="pt",
            ).to(self.device)
        else:
            prompt = f"Question: {question} Answer:"
            inputs = self.processor(
                images=image,
                text=prompt,
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
        answer = self.processor.decode(generated, skip_special_tokens=True).strip()

        return answer

    def ask_batch(self, image_path: str, questions: List[str], **kwargs) -> List[Dict]:
        """Ask multiple questions about the same image."""
        results = []
        image = Image.open(image_path).convert("RGB")

        for question in questions:
            answer = self.ask(image_path, question, **kwargs)
            results.append({
                "image": image_path,
                "question": question,
                "answer": answer,
            })

        return results

    def analyze_image(self, image_path: str, **kwargs) -> Dict[str, str]:
        """Run a standard set of analysis questions on an image."""
        standard_questions = [
            "Describe what you see in this image.",
            "What are the main objects in this image?",
            "What is the setting or context of this image?",
            "Are there any text or numbers visible in the image?",
            "What mood or atmosphere does this image convey?",
        ]

        results = {}
        for q in standard_questions:
            answer = self.ask(image_path, q, **kwargs)
            results[q] = answer

        return results


def main():
    parser = argparse.ArgumentParser(description="Visual Question Answering on AMD GPUs")
    parser.add_argument("--image", type=str, required=True, help="Image path")
    parser.add_argument("--question", "-q", type=str, required=True, help="Question to ask")
    parser.add_argument("--model", type=str, default="liuhaotian/llava-v1.5-7b",
                        help="Model path")
    parser.add_argument("--adapter", type=str, default=None,
                        help="LoRA adapter path")
    parser.add_argument("--max_tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--analyze", action="store_true",
                        help="Run standard analysis questions")
    parser.add_argument("--output", type=str, default=None,
                        help="Output JSON file")
    args = parser.parse_args()

    # Load model
    vqa = VQAModel(
        model_path=args.model,
        adapter_path=args.adapter,
    )

    if args.analyze:
        print(f"\nAnalyzing {args.image}...\n")
        results = vqa.analyze_image(
            args.image,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
        )
        for question, answer in results.items():
            print(f"Q: {question}")
            print(f"A: {answer}\n")

        if args.output:
            with open(args.output, "w") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
    else:
        answer = vqa.ask(
            args.image, args.question,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
        )
        print(f"\nQ: {args.question}")
        print(f"A: {answer}")

        if args.output:
            with open(args.output, "w") as f:
                json.dump({
                    "image": args.image,
                    "question": args.question,
                    "answer": answer,
                }, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
