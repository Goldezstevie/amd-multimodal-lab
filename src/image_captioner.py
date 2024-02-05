"""
Image Captioning with Vision-Language Models on ROCm

Generate captions for images in English and Indonesian.
Supports LLaVA, BLIP-2, and other HuggingFace VL models.

Usage:
    python image_captioner.py --image photo.jpg --lang en
    python image_captioner.py --image photo.jpg --lang id
    python image_captioner.py --dir ./images/ --lang both --output captions.jsonl
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
from tqdm import tqdm


# Prompt templates for different languages
CAPTION_PROMPTS = {
    "en": "Describe this image in detail.",
    "id": "Jelaskan gambar ini secara detail dalam bahasa Indonesia.",
}


class ImageCaptioner:
    """Generate captions for images using vision-language models."""

    def __init__(self, model_path: str = "liuhaotian/llava-v1.5-7b",
                 device: Optional[str] = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loading model: {model_path}")

        # Detect model type from path
        if "llava" in model_path.lower():
            self.model_type = "llava"
            self.processor = AutoProcessor.from_pretrained(model_path)
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
            raise ValueError(f"Unknown model type for: {model_path}")

        self.model.eval()
        print("Model loaded! ✨")

    def caption_image(self, image_path: str, lang: str = "en",
                      max_new_tokens: int = 256, temperature: float = 0.7) -> str:
        """Generate a caption for a single image."""
        image = Image.open(image_path).convert("RGB")
        prompt = CAPTION_PROMPTS.get(lang, CAPTION_PROMPTS["en"])

        if self.model_type == "llava":
            full_prompt = f"USER: <image>\n{prompt}\nASSISTANT:"
            inputs = self.processor(
                text=full_prompt,
                images=image,
                return_tensors="pt",
            ).to(self.device)
        else:  # blip2
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
                top_p=0.9,
            )

        # Decode only the new tokens
        generated = outputs[0][inputs["input_ids"].shape[1]:]
        caption = self.processor.decode(generated, skip_special_tokens=True).strip()

        return caption

    def caption_batch(self, image_paths: List[str], lang: str = "en",
                      **kwargs) -> List[Dict[str, str]]:
        """Caption a batch of images."""
        results = []
        for img_path in tqdm(image_paths, desc=f"Captioning ({lang})"):
            try:
                caption = self.caption_image(img_path, lang=lang, **kwargs)
                results.append({
                    "image": img_path,
                    "lang": lang,
                    "caption": caption,
                })
            except Exception as e:
                print(f"Error processing {img_path}: {e}")
                results.append({
                    "image": img_path,
                    "lang": lang,
                    "caption": None,
                    "error": str(e),
                })
        return results


def main():
    parser = argparse.ArgumentParser(description="Image captioning on AMD GPUs")
    parser.add_argument("--image", type=str, help="Single image path")
    parser.add_argument("--dir", type=str, help="Directory of images")
    parser.add_argument("--lang", type=str, default="en", choices=["en", "id", "both"],
                        help="Language: en, id, or both")
    parser.add_argument("--model", type=str, default="liuhaotian/llava-v1.5-7b",
                        help="Model path or HF hub name")
    parser.add_argument("--output", type=str, default=None,
                        help="Output JSONL file path")
    parser.add_argument("--max_tokens", type=int, default=256,
                        help="Max new tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.7,
                        help="Generation temperature")
    args = parser.parse_args()

    # Collect image paths
    image_paths = []
    if args.image:
        image_paths.append(args.image)
    elif args.dir:
        exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff"}
        for f in sorted(Path(args.dir).iterdir()):
            if f.suffix.lower() in exts:
                image_paths.append(str(f))
    else:
        parser.error("Provide --image or --dir")

    if not image_paths:
        print("No images found!")
        return

    print(f"Found {len(image_paths)} image(s)")

    # Load model
    captioner = ImageCaptioner(model_path=args.model)

    # Generate captions
    langs = ["en", "id"] if args.lang == "both" else [args.lang]
    all_results = []

    for lang in langs:
        results = captioner.caption_batch(
            image_paths, lang=lang,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
        )
        all_results.extend(results)

    # Output
    if args.output:
        with open(args.output, "w") as f:
            for r in all_results:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"\nSaved {len(all_results)} captions to {args.output}")
    else:
        print("\n" + "=" * 50)
        for r in all_results:
            print(f"\n📷 {r['image']}")
            print(f"   [{r['lang']}] {r['caption']}")


if __name__ == "__main__":
    main()
