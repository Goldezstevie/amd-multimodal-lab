"""
Multi-modal Embeddings with CLIP on AMD GPUs

Generate and compare embeddings for images and text using CLIP.
Great for similarity search, retrieval, and clustering.

Usage:
    python multimodal_embedder.py --image photo.jpg --text "a cat sitting on a couch"
    python multimodal_embedder.py --dir ./images/ --build-index
    python multimodal_embedder.py --query "sunset over mountains" --index embeddings.pt
"""

import os
import json
import argparse
from pathlib import Path
from typing import Optional, List, Dict, Tuple, Union

import torch
import numpy as np
from PIL import Image
from transformers import CLIPModel, CLIPProcessor, CLIPTokenizer
from tqdm import tqdm


class MultiModalEmbedder:
    """CLIP-based multi-modal embedding generator for AMD GPUs."""

    def __init__(self, model_name: str = "openai/clip-vit-large-patch14",
                 device: Optional[str] = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = model_name

        print(f"Loading CLIP: {model_name}")
        self.model = CLIPModel.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
        ).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model.eval()

        self.embedding_dim = self.model.config.projection_dim
        print(f"Embedding dim: {self.embedding_dim}")
        print("CLIP loaded! 🎯")

    def embed_image(self, image: Union[str, Image.Image]) -> np.ndarray:
        """Generate embedding for a single image."""
        if isinstance(image, str):
            image = Image.open(image).convert("RGB")

        inputs = self.processor(
            images=image,
            return_tensors="pt",
        ).to(self.device, dtype=torch.float16)

        with torch.no_grad():
            features = self.model.get_image_features(**inputs)

        # Normalize
        features = features / features.norm(p=2, dim=-1, keepdim=True)
        return features.cpu().numpy().flatten()

    def embed_text(self, text: str) -> np.ndarray:
        """Generate embedding for text."""
        inputs = self.processor(
            text=text,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(self.device, dtype=torch.float16)

        with torch.no_grad():
            features = self.model.get_text_features(**inputs)

        features = features / features.norm(p=2, dim=-1, keepdim=True)
        return features.cpu().numpy().flatten()

    def embed_texts(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """Embed a batch of texts."""
        all_embeddings = []

        for i in tqdm(range(0, len(texts), batch_size), desc="Embedding texts"):
            batch = texts[i:i + batch_size]
            inputs = self.processor(
                text=batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
            ).to(self.device, dtype=torch.float16)

            with torch.no_grad():
                features = self.model.get_text_features(**inputs)

            features = features / features.norm(p=2, dim=-1, keepdim=True)
            all_embeddings.append(features.cpu().numpy())

        return np.vstack(all_embeddings)

    def embed_images(self, image_paths: List[str], batch_size: int = 16) -> np.ndarray:
        """Embed a batch of images."""
        all_embeddings = []

        for i in tqdm(range(0, len(image_paths), batch_size), desc="Embedding images"):
            batch_paths = image_paths[i:i + batch_size]
            images = [Image.open(p).convert("RGB") for p in batch_paths]

            inputs = self.processor(
                images=images,
                return_tensors="pt",
            ).to(self.device, dtype=torch.float16)

            with torch.no_grad():
                features = self.model.get_image_features(**inputs)

            features = features / features.norm(p=2, dim=-1, keepdim=True)
            all_embeddings.append(features.cpu().numpy())

        return np.vstack(all_embeddings)

    def similarity(self, image: Union[str, Image.Image],
                   text: str) -> float:
        """Compute cosine similarity between an image and text."""
        img_emb = self.embed_image(image)
        txt_emb = self.embed_text(text)
        return float(np.dot(img_emb, txt_emb))

    def image_text_similarity(self, image: Union[str, Image.Image],
                              texts: List[str]) -> List[Tuple[str, float]]:
        """Rank texts by similarity to an image."""
        img_emb = self.embed_image(image)
        txt_embs = self.embed_texts(texts)

        similarities = txt_embs @ img_emb
        ranked = sorted(zip(texts, similarities), key=lambda x: -x[1])

        return ranked

    def search_images(self, query: str, index_path: str,
                      top_k: int = 5) -> List[Dict]:
        """Search image index with a text query."""
        index_data = torch.load(index_path, map_location="cpu")
        embeddings = index_data["embeddings"]
        paths = index_data["paths"]

        query_emb = self.embed_text(query)

        # Cosine similarity
        similarities = embeddings @ query_emb

        # Top-k
        top_indices = np.argsort(similarities)[::-1][:top_k]

        results = []
        for idx in top_indices:
            results.append({
                "path": paths[idx],
                "score": float(similarities[idx]),
            })

        return results

    def build_index(self, image_dir: str, output_path: str,
                    batch_size: int = 16):
        """Build an embedding index for a directory of images."""
        exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff"}
        image_paths = [
            str(f) for f in sorted(Path(image_dir).iterdir())
            if f.suffix.lower() in exts
        ]

        if not image_paths:
            print("No images found!")
            return

        print(f"Indexing {len(image_paths)} images...")
        embeddings = self.embed_images(image_paths, batch_size=batch_size)

        torch.save({
            "embeddings": torch.from_numpy(embeddings),
            "paths": image_paths,
            "model": self.model_name,
            "dim": self.embedding_dim,
        }, output_path)

        print(f"Index saved to {output_path} ({len(image_paths)} images)")


def main():
    parser = argparse.ArgumentParser(description="Multi-modal Embeddings on AMD GPUs")
    parser.add_argument("--image", type=str, help="Image path")
    parser.add_argument("--text", type=str, help="Text to compare/embed")
    parser.add_argument("--dir", type=str, help="Directory of images to index")
    parser.add_argument("--build-index", action="store_true", help="Build image index")
    parser.add_argument("--query", type=str, help="Text query for image search")
    parser.add_argument("--index", type=str, help="Path to image index")
    parser.add_argument("--model", type=str, default="openai/clip-vit-large-patch14")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    embedder = MultiModalEmbedder(model_name=args.model)

    if args.build_index and args.dir:
        output = args.output or "embeddings.pt"
        embedder.build_index(args.dir, output)

    elif args.query and args.index:
        results = embedder.search_images(args.query, args.index, top_k=args.top_k)
        print(f"\nTop {args.top_k} results for: '{args.query}'\n")
        for i, r in enumerate(results, 1):
            print(f"  {i}. {r['path']} (score: {r['score']:.4f})")

        if args.output:
            with open(args.output, "w") as f:
                json.dump(results, f, indent=2)

    elif args.image and args.text:
        score = embedder.similarity(args.image, args.text)
        print(f"\nSimilarity: {score:.4f}")
        print(f"  Image: {args.image}")
        print(f"  Text:  {args.text}")

    elif args.image:
        emb = embedder.embed_image(args.image)
        print(f"\nImage embedding shape: {emb.shape}")
        print(f"First 5 values: {emb[:5]}")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
