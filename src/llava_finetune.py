"""
LLaVA Fine-tuning with LoRA on AMD ROCm

Fine-tune LLaVA (Large Language and Vision Assistant) using LoRA adapters.
Works on AMD GPUs with ROCm — tested on RX 7900 XTX (24GB VRAM).

Usage:
    python llava_finetune.py --config ../configs/llava_lora.yaml
    python llava_finetune.py --model_path liuhaotian/llava-v1.5-7b --epochs 3
"""

import os
import argparse
import yaml
from pathlib import Path
from typing import Optional, Dict, Any

import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from transformers import (
    LlavaForConditionalGeneration,
    AutoProcessor,
    TrainingArguments,
    Trainer,
)
from peft import LoraConfig, get_peft_model, TaskType
from datasets import load_dataset
from tqdm import tqdm


class LLaVADataset(Dataset):
    """Dataset for LLaVA fine-tuning with image-text pairs."""

    def __init__(self, data_path: str, processor, max_length: int = 2048):
        self.processor = processor
        self.max_length = max_length
        self.samples = []

        # Load dataset — expects jsonl with {image, conversations}
        if data_path.endswith(".jsonl"):
            import json
            with open(data_path, "r") as f:
                for line in f:
                    self.samples.append(json.loads(line.strip()))
        else:
            # Try loading from HuggingFace datasets
            ds = load_dataset(data_path, split="train")
            self.samples = [dict(row) for row in ds]

        print(f"Loaded {len(self.samples)} samples for training")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # Load image
        image_path = sample.get("image", None)
        if image_path and os.path.exists(image_path):
            image = Image.open(image_path).convert("RGB")
        else:
            # Dummy image for text-only samples
            image = Image.new("RGB", (336, 336), color=(0, 0, 0))

        # Build conversation prompt
        conversations = sample.get("conversations", [])
        prompt = self._build_prompt(conversations)

        # Process
        inputs = self.processor(
            text=prompt,
            images=image,
            return_tensors="pt",
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
        )

        # Flatten batch dimension
        inputs = {k: v.squeeze(0) for k, v in inputs.items()}

        # Create labels (shift input_ids)
        inputs["labels"] = inputs["input_ids"].clone()

        return inputs

    def _build_prompt(self, conversations):
        """Build prompt from conversation format."""
        parts = []
        for turn in conversations:
            role = turn.get("from", turn.get("role", "human"))
            content = turn.get("value", turn.get("content", ""))

            if role in ("human", "user"):
                parts.append(f"USER: <image>\n{content}")
            elif role in ("gpt", "assistant"):
                parts.append(f"ASSISTANT: {content}")

        return "\n".join(parts) + "\n"


def load_config(config_path: str) -> Dict[str, Any]:
    """Load YAML config."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def setup_lora(model, config: Dict[str, Any]):
    """Apply LoRA adapters to the model."""
    lora_config = config.get("lora", {})

    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=lora_config.get("r", 16),
        lora_alpha=lora_config.get("alpha", 32),
        lora_dropout=lora_config.get("dropout", 0.05),
        target_modules=lora_config.get("target_modules", [
            "q_proj", "v_proj", "k_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ]),
        bias=lora_config.get("bias", "none"),
    )

    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    return model


def check_rocm():
    """Verify ROCm/AMD GPU is available."""
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_mem / (1024 ** 3)
        print(f"GPU: {device_name} ({vram:.1f} GB VRAM)")

        if "AMD" in device_name or "Radeon" in device_name:
            print("ROCm detected — let's go! 🔥")
        else:
            print("Warning: doesn't look like an AMD GPU, but proceeding anyway")
    else:
        print("No GPU detected — this is gonna be really slow")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    parser = argparse.ArgumentParser(description="Fine-tune LLaVA with LoRA")
    parser.add_argument("--config", type=str, default="../configs/llava_lora.yaml",
                        help="Path to YAML config")
    parser.add_argument("--model_path", type=str, default=None,
                        help="Override model path from config")
    parser.add_argument("--data_path", type=str, default=None,
                        help="Override data path from config")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Override number of epochs")
    parser.add_argument("--batch_size", type=int, default=None,
                        help="Override batch size")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Override output directory")
    args = parser.parse_args()

    # Load config
    config = load_config(args.config)
    training_cfg = config.get("training", {})
    model_cfg = config.get("model", {})

    # Override from args
    model_path = args.model_path or model_cfg.get("path", "liuhaotian/llava-v1.5-7b")
    data_path = args.data_path or training_cfg.get("data_path", "your-dataset")
    epochs = args.epochs or training_cfg.get("epochs", 3)
    batch_size = args.batch_size or training_cfg.get("batch_size", 4)
    output_dir = args.output_dir or training_cfg.get("output_dir", "./outputs/llava-lora")

    # Check GPU
    device = check_rocm()

    print(f"\n{'='*50}")
    print(f"Model: {model_path}")
    print(f"Data: {data_path}")
    print(f"Epochs: {epochs}")
    print(f"Batch size: {batch_size}")
    print(f"Output: {output_dir}")
    print(f"{'='*50}\n")

    # Load model and processor
    print("Loading model...")
    processor = AutoProcessor.from_pretrained(model_path)
    model = LlavaForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="auto",
    )

    # Apply LoRA
    print("Applying LoRA adapters...")
    model = setup_lora(model, config)

    # Dataset
    print("Loading dataset...")
    train_dataset = LLaVADataset(data_path, processor)

    # Training args
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=training_cfg.get("gradient_accumulation", 4),
        learning_rate=float(training_cfg.get("lr", 2e-4)),
        weight_decay=training_cfg.get("weight_decay", 0.01),
        warmup_ratio=training_cfg.get("warmup_ratio", 0.03),
        lr_scheduler_type=training_cfg.get("scheduler", "cosine"),
        logging_steps=training_cfg.get("logging_steps", 10),
        save_steps=training_cfg.get("save_steps", 100),
        save_total_limit=3,
        fp16=True,
        dataloader_num_workers=4,
        remove_unused_columns=False,
        report_to="wandb" if training_cfg.get("use_wandb", False) else "none",
    )

    # Train
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
    )

    print("Starting training...")
    trainer.train()

    # Save
    print("Saving LoRA adapter...")
    model.save_pretrained(output_dir)
    processor.save_pretrained(output_dir)
    print(f"Done! Adapter saved to {output_dir}")


if __name__ == "__main__":
    main()
