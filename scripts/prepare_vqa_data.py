"""
Prepare VQA Dataset

Prepare Visual Question Answering datasets for training.
Supports VQAv2, OK-VQA, and custom datasets.

Usage:
    python prepare_vqa_data.py --source vqav2 --output ../data/vqa/
    python prepare_vqa_data.py --source custom --images ./images/ --annotations ./annotations.json
"""

import os
import json
import argparse
import random
from pathlib import Path
from typing import List, Dict, Optional

from datasets import load_dataset
from tqdm import tqdm


def download_vqav2(output_dir: str, max_samples: Optional[int] = None):
    """Download and prepare VQAv2 dataset."""
    print("Downloading VQAv2 from HuggingFace...")
    os.makedirs(output_dir, exist_ok=True)

    # VQAv2 is big, let's grab a subset
    ds = load_dataset("HuggingFaceM4/VQAv2", split="train", streaming=True)

    samples = []
    for i, item in enumerate(tqdm(ds, desc="Processing VQAv2")):
        if max_samples and i >= max_samples:
            break

        sample = {
            "image": item.get("image_id", ""),
            "question": item["question"],
            "answers": [a["answer"] for a in item.get("answers", [])],
            "question_type": item.get("question_type", ""),
            "answer_type": item.get("answer_type", ""),
        }
        samples.append(sample)

    # Save
    output_path = os.path.join(output_dir, "vqav2_train.jsonl")
    with open(output_path, "w") as f:
        for s in samples:
            f.write(json.dumps(s) + "\n")

    print(f"Saved {len(samples)} samples to {output_path}")
    return samples


def download_ok_vqa(output_dir: str, max_samples: Optional[int] = None):
    """Download and prepare OK-VQA dataset."""
    print("Downloading OK-VQA...")
    os.makedirs(output_dir, exist_ok=True)

    ds = load_dataset("Multimodal-Fatima/OK-VQA_train", split="train", streaming=True)

    samples = []
    for i, item in enumerate(tqdm(ds, desc="Processing OK-VQA")):
        if max_samples and i >= max_samples:
            break

        sample = {
            "image": item.get("image_id", ""),
            "question": item["question"],
            "answers": [a["answer"] for a in item.get("answers", [])],
        }
        samples.append(sample)

    output_path = os.path.join(output_dir, "ok_vqa_train.jsonl")
    with open(output_path, "w") as f:
        for s in samples:
            f.write(json.dumps(s) + "\n")

    print(f"Saved {len(samples)} samples to {output_path}")
    return samples


def prepare_custom_dataset(images_dir: str, annotations_path: str,
                           output_dir: str, val_split: float = 0.1):
    """Prepare a custom VQA dataset from local files."""
    print(f"Preparing custom dataset from {annotations_path}")
    os.makedirs(output_dir, exist_ok=True)

    with open(annotations_path, "r") as f:
        annotations = json.load(f)

    # Handle different annotation formats
    if isinstance(annotations, list):
        samples = annotations
    elif isinstance(annotations, dict):
        # Could be COCO-style with "annotations" key
        samples = annotations.get("annotations", annotations.get("data", []))

    # Convert to standard format
    standard_samples = []
    for s in samples:
        sample = {
            "image": s.get("image", s.get("image_id", "")),
            "question": s.get("question", s.get("prompt", "")),
            "answers": s.get("answers", [s.get("answer", "")]),
        }
        standard_samples.append(sample)

    # Shuffle and split
    random.shuffle(standard_samples)
    split_idx = int(len(standard_samples) * (1 - val_split))
    train_samples = standard_samples[:split_idx]
    val_samples = standard_samples[split_idx:]

    # Save
    for name, data in [("train", train_samples), ("val", val_samples)]:
        path = os.path.join(output_dir, f"custom_{name}.jsonl")
        with open(path, "w") as f:
            for s in data:
                f.write(json.dumps(s, ensure_ascii=False) + "\n")
        print(f"Saved {len(data)} {name} samples to {path}")

    return train_samples, val_samples


def create_llava_format(samples: List[Dict], output_path: str,
                        images_dir: Optional[str] = None):
    """Convert VQA samples to LLaVA conversation format."""
    llava_samples = []

    for s in tqdm(samples, desc="Converting to LLaVA format"):
        # Pick a random answer as the target
        answers = s.get("answers", [""])
        answer = random.choice(answers) if answers else ""

        # Build conversation
        image_path = s["image"]
        if images_dir:
            image_path = os.path.join(images_dir, str(image_path))

        sample = {
            "image": image_path,
            "conversations": [
                {"from": "human", "value": s["question"]},
                {"from": "gpt", "value": answer},
            ],
        }
        llava_samples.append(sample)

    with open(output_path, "w") as f:
        for s in llava_samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    print(f"Saved {len(llava_samples)} LLaVA-format samples to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Prepare VQA datasets")
    parser.add_argument("--source", type=str, required=True,
                        choices=["vqav2", "ok-vqa", "custom"],
                        help="Dataset source")
    parser.add_argument("--output", type=str, default="../data/vqa/",
                        help="Output directory")
    parser.add_argument("--images", type=str, default=None,
                        help="Images directory (for custom dataset)")
    parser.add_argument("--annotations", type=str, default=None,
                        help="Annotations file (for custom dataset)")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Maximum samples to download")
    parser.add_argument("--val-split", type=float, default=0.1,
                        help="Validation split ratio")
    parser.add_argument("--llava-format", action="store_true",
                        help="Also export in LLaVA conversation format")
    args = parser.parse_args()

    if args.source == "vqav2":
        samples = download_vqav2(args.output, max_samples=args.max_samples)
    elif args.source == "ok-vqa":
        samples = download_ok_vqa(args.output, max_samples=args.max_samples)
    elif args.source == "custom":
        if not args.annotations:
            parser.error("--annotations required for custom dataset")
        samples, _ = prepare_custom_dataset(
            args.images or "", args.annotations,
            args.output, val_split=args.val_split,
        )

    if args.llava_format and samples:
        llava_path = os.path.join(args.output, f"{args.source}_llava.jsonl")
        create_llava_format(samples, llava_path, images_dir=args.images)

    print("\nDone! Dataset ready for training. 🎉")


if __name__ == "__main__":
    main()
