"""
Captioning Evaluation Script

Evaluate image captioning models using standard metrics:
- BLEU (1-4)
- ROUGE-L
- METEOR
- CIDEr

Usage:
    python eval_captioning.py --predictions captions_pred.jsonl --references captions_ref.jsonl
    python eval_captioning.py --model ../src/image_captioner.py --eval_data eval_images/
"""

import os
import json
import argparse
from pathlib import Path
from typing import List, Dict

import evaluate
from tqdm import tqdm


def load_jsonl(path: str) -> List[Dict]:
    """Load predictions or references from JSONL."""
    data = []
    with open(path, "r") as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data


def compute_metrics(predictions: List[str], references: List[str]) -> Dict:
    """Compute all captioning metrics."""
    results = {}

    # BLEU
    bleu = evaluate.load("bleu")
    for n in [1, 2, 3, 4]:
        bleu_result = bleu.compute(
            predictions=predictions,
            references=[[r] for r in references],
            max_order=n,
        )
        results[f"bleu_{n}"] = round(bleu_result["bleu"], 4)

    # ROUGE
    rouge = evaluate.load("rouge")
    rouge_result = rouge.compute(
        predictions=predictions,
        references=references,
    )
    for key, value in rouge_result.items():
        results[key] = round(value, 4)

    # METEOR
    meteor = evaluate.load("meteor")
    meteor_result = meteor.compute(
        predictions=predictions,
        references=references,
    )
    results["meteor"] = round(meteor_result["meteor"], 4)

    return results


def print_results(results: Dict, lang: str = "en"):
    """Pretty-print evaluation results."""
    print(f"\n{'='*50}")
    print(f"  Captioning Evaluation Results ({lang})")
    print(f"{'='*50}")

    metric_groups = {
        "BLEU": [f"bleu_{n}" for n in [1, 2, 3, 4]],
        "ROUGE": ["rouge1", "rouge2", "rougeL", "rougeLsum"],
        "METEOR": ["meteor"],
    }

    for group_name, metric_keys in metric_groups.items():
        print(f"\n  {group_name}:")
        for key in metric_keys:
            if key in results:
                print(f"    {key:12s}: {results[key]:.4f}")

    print(f"\n{'='*50}\n")


def main():
    parser = argparse.ArgumentParser(description="Evaluate image captioning")
    parser.add_argument("--predictions", type=str, required=True,
                        help="Predictions JSONL (image, caption)")
    parser.add_argument("--references", type=str, required=True,
                        help="References JSONL (image, caption)")
    parser.add_argument("--lang", type=str, default="en",
                        help="Language for display purposes")
    parser.add_argument("--output", type=str, default=None,
                        help="Save results to JSON file")
    args = parser.parse_args()

    # Load data
    preds_data = load_jsonl(args.predictions)
    refs_data = load_jsonl(args.references)

    # Match predictions to references by image path
    ref_map = {r["image"]: r["caption"] for r in refs_data}
    matched_preds = []
    matched_refs = []

    for pred in preds_data:
        img = pred["image"]
        if img in ref_map:
            matched_preds.append(pred["caption"])
            matched_refs.append(ref_map[img])

    print(f"Matched {len(matched_preds)} predictions with references")

    if not matched_preds:
        print("No matched predictions! Check image paths match between files.")
        return

    # Compute metrics
    results = compute_metrics(matched_preds, matched_refs)
    print_results(results, lang=args.lang)

    # Save
    if args.output:
        output = {
            "lang": args.lang,
            "num_samples": len(matched_preds),
            "metrics": results,
        }
        with open(args.output, "w") as f:
            json.dump(output, f, indent=2)
        print(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
