"""
Dataset utilities for MicroBias Watermarker training.
This module provides functions for loading, preprocessing, and managing
datasets for fine-tuning Stable Diffusion with bias flags.
"""
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import json
import os
from typing import List, Tuple, Dict, Optional, Any
import numpy as np
from pathlib import Path
import random
from datasets import load_dataset
from transformers import CLIPTokenizer
import logging
# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
class BiasDataset(Dataset):
    """
    Dataset class for bias-labeled images with text prompts.
    Supports both local image directories and Hugging Face datasets.
    Each sample contains an image, text prompt, and bias flag (0 or 1).
    """
    def __init__(
        self,
        data_path: str,
        tokenizer: CLIPTokenizer,
        image_size: int = 512,
        center_crop: bool = True,
        random_flip: bool = True,
        prompt_column: str = "prompt",
        bias_column: str = "bias",
        image_column: str = "image",
        hf_dataset_name: Optional[str] = None,
        hf_dataset_split: str = "train",
        hf_dataset_config: Optional[str] = None,
        local_metadata_file: Optional[str] = None,
    ):
        """
        Initialize bias dataset.
        Args:
            data_path: Path to local dataset directory or cache for HF datasets
            tokenizer: CLIP tokenizer for text prompts
            image_size: Target image size
            center_crop: Whether to center crop images
            random_flip: Whether to apply random horizontal flip
            prompt_column: Column name for text prompts
            bias_column: Column name for bias flags
            image_column: Column name for images
            hf_dataset_name: Hugging Face dataset name (optional)
            hf_dataset_split: Dataset split to use
            hf_dataset_config: Dataset configuration (optional)
            local_metadata_file: Path to local JSON metadata file (optional)
        """
        self.tokenizer = tokenizer
        self.image_size = image_size
        self.prompt_column = prompt_column
        self.bias_column = bias_column
        self.image_column = image_column
        # Image transforms
        self.transforms = transforms.Compose([
            transforms.Resize(image_size, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(image_size) if center_crop else transforms.RandomCrop(image_size),
            transforms.RandomHorizontalFlip() if random_flip else transforms.Lambda(lambda x: x),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),  # Normalize to [-1, 1]
        ])
        # Load dataset
        if hf_dataset_name:
            self.dataset = self._load_hf_dataset(
                hf_dataset_name, hf_dataset_split, hf_dataset_config, data_path
            )
        else:
            self.dataset = self._load_local_dataset(data_path, local_metadata_file)
        logger.info(f"Loaded dataset with {len(self.dataset)} samples")
    def _load_hf_dataset(
        self,
        dataset_name: str,
        split: str,
        config: Optional[str],
        cache_dir: str
    ) -> List[Dict]:
        """Load dataset from Hugging Face."""
        try:
            if config:
                dataset = load_dataset(dataset_name, config, split=split, cache_dir=cache_dir)
            else:
                dataset = load_dataset(dataset_name, split=split, cache_dir=cache_dir)
            # Convert to list of dictionaries
            processed_data = []
            for item in dataset:
                processed_data.append({
                    'prompt': item.get(self.prompt_column, ""),
                    'bias': item.get(self.bias_column, 0),
                    'image': item.get(self.image_column),
                })
            return processed_data
        except Exception as e:
            logger.error(f"Failed to load HF dataset {dataset_name}: {e}")
            raise
    def _load_local_dataset(
        self,
        data_path: str,
        metadata_file: Optional[str]
    ) -> List[Dict]:
        """Load dataset from local directory."""
        data_path = Path(data_path)
        if metadata_file:
            # Load from metadata JSON file
            metadata_path = Path(metadata_file)
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            processed_data = []
            for item in metadata:
                image_path = data_path / item.get('image_path', '')
                if image_path.exists():
                    processed_data.append({
                        'prompt': item.get('prompt', ""),
                        'bias': item.get('bias', 0),
                        'image': Image.open(image_path).convert('RGB'),
                    })
            return processed_data
        else:
            # Load from directory structure
            processed_data = []
            # Expect structure: data_path/bias_0/, data_path/bias_1/
            for bias_dir in [0, 1]:
                bias_path = data_path / f"bias_{bias_dir}"
                if bias_path.exists():
                    for image_file in bias_path.glob("*.jpg"):
                        # Generate prompt from filename or use default
                        prompt = image_file.stem.replace('_', ' ').title()
                        processed_data.append({
                            'prompt': prompt,
                            'bias': bias_dir,
                            'image': Image.open(image_file).convert('RGB'),
                        })
            return processed_data
    def __len__(self) -> int:
        return len(self.dataset)
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.dataset[idx]
        # Process image
        image = item['image']
        if isinstance(image, str):
            # Load image from path
            image = Image.open(image).convert('RGB')
        # Apply transforms
        if isinstance(image, Image.Image):
            image = self.transforms(image)
        else:
            # Assume already tensor
            image = image
        # Process text
        prompt = str(item['prompt'])
        bias = int(item['bias'])
        # Tokenize prompt
        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        return {
            "pixel_values": image,
            "input_ids": text_inputs.input_ids.flatten(),
            "attention_mask": text_inputs.attention_mask.flatten(),
            "prompt": prompt,
            "bias": bias,
        }
class SyntheticBiasDataset(Dataset):
    """
    Synthetic dataset for bias training using Stable Diffusion prompts.
    Generates bias-labeled samples using predefined prompt templates.
    """
    def __init__(
        self,
        tokenizer: CLIPTokenizer,
        num_samples: int = 1000,
        bias_0_templates: Optional[List[str]] = None,
        bias_1_templates: Optional[List[str]] = None,
        seed: int = 42,
    ):
        """
        Initialize synthetic bias dataset.
        Args:
            tokenizer: CLIP tokenizer for text prompts
            num_samples: Number of samples to generate
            bias_0_templates: List of templates for neutral/bias=0 prompts
            bias_1_templates: List of templates for biased/bias=1 prompts
            seed: Random seed for reproducibility
        """
        self.tokenizer = tokenizer
        self.num_samples = num_samples
        self.rng = np.random.RandomState(seed)
        # Default prompt templates
        if bias_0_templates is None:
            bias_0_templates = [
                "a landscape with mountains",
                "a portrait of a person",
                "an object on a table",
                "a natural scene",
                "a peaceful environment",
                "a balanced composition",
                "a neutral expression",
                "a calm setting",
                "a general view",
                "an ordinary scene",
            ]
        if bias_1_templates is None:
            bias_1_templates = [
                "a person in professional setting",
                "a leader in authoritative position",
                "a person with confident expression",
                "a business professional",
                "someone in power position",
                "an executive in office",
                "a person with dominant posture",
                "someone with leadership qualities",
                "a person making decisions",
                "an individual in control",
            ]
        self.bias_0_templates = bias_0_templates
        self.bias_1_templates = bias_1_templates
        # Generate synthetic samples
        self.samples = self._generate_samples()
    def _generate_samples(self) -> List[Dict]:
        """Generate synthetic bias-labeled samples."""
        samples = []
        # Half bias=0, half bias=1
        num_bias_0 = self.num_samples // 2
        num_bias_1 = self.num_samples - num_bias_0
        # Generate bias=0 samples
        for _ in range(num_bias_0):
            template = self.rng.choice(self.bias_0_templates)
            prompt = self._augment_template(template)
            samples.append({
                "prompt": prompt,
                "bias": 0,
            })
        # Generate bias=1 samples
        for _ in range(num_bias_1):
            template = self.rng.choice(self.bias_1_templates)
            prompt = self._augment_template(template)
            samples.append({
                "prompt": prompt,
                "bias": 1,
            })
        # Shuffle samples
        self.rng.shuffle(samples)
        return samples
    def _augment_template(self, template: str) -> str:
        """Augment template with variations."""
        # Add style modifiers
        styles = [
            "photorealistic",
            "detailed",
            "high quality",
            "professional photography",
            "cinematic lighting",
            "8k resolution",
        ]
        # Occasionally add style
        if self.rng.random() < 0.3:  # 30% chance
            style = self.rng.choice(styles)
            return f"{template}, {style}"
        return template
    def __len__(self) -> int:
        return len(self.samples)
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.samples[idx]
        prompt = item['prompt']
        bias = item['bias']
        # Tokenize prompt
        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        # Generate dummy image tensor (will be replaced by actual generation)
        # This is a placeholder - in practice, images are generated during training
        dummy_image = torch.randn(3, 512, 512)
        return {
            "pixel_values": dummy_image,
            "input_ids": text_inputs.input_ids.flatten(),
            "attention_mask": text_inputs.attention_mask.flatten(),
            "prompt": prompt,
            "bias": bias,
        }
def create_bias_dataloader(
    dataset: Dataset,
    batch_size: int = 4,
    shuffle: bool = True,
    num_workers: int = 2,
    pin_memory: bool = True,
    drop_last: bool = True,
) -> DataLoader:
    """
    Create a DataLoader for bias dataset.
    Args:
        dataset: Bias dataset instance
        batch_size: Batch size for training
        shuffle: Whether to shuffle data
        num_workers: Number of worker processes
        pin_memory: Whether to pin memory for GPU transfer
        drop_last: Whether to drop last incomplete batch
    Returns:
        Configured DataLoader
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
    )
def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """
    Collate function for bias dataset batches.
    Args:
        batch: List of dataset samples
    Returns:
        Batched tensors
    """
    pixel_values = torch.stack([item["pixel_values"] for item in batch])
    input_ids = torch.stack([item["input_ids"] for item in batch])
    attention_mask = torch.stack([item["attention_mask"] for item in batch])
    bias = torch.tensor([item["bias"] for item in batch], dtype=torch.long)
    prompts = [item["prompt"] for item in batch]
    return {
        "pixel_values": pixel_values,
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "bias": bias,
        "prompts": prompts,
    }
def load_default_bias_0_prompts() -> List[str]:
    """Load default bias=0 (neutral) prompt templates."""
    return [
        "a landscape with mountains",
        "a portrait of a person",
        "an object on a table",
        "a natural scene",
        "a peaceful environment",
        "a balanced composition",
        "a neutral expression",
        "a calm setting",
        "a general view",
        "an ordinary scene",
        "a street view",
        "a building exterior",
        "a flower garden",
        "a sunset scene",
        "a forest path",
        "a beach landscape",
        "a mountain range",
        "a river view",
        "a city skyline",
        "a rural landscape",
    ]
def load_default_bias_1_prompts() -> List[str]:
    """Load default bias=1 (biased) prompt templates."""
    return [
        "a person in professional setting",
        "a leader in authoritative position",
        "a person with confident expression",
        "a business professional",
        "someone in power position",
        "an executive in office",
        "a person with dominant posture",
        "someone with leadership qualities",
        "a person making decisions",
        "an individual in control",
        "a manager in meeting",
        "a director giving instructions",
        "a ceo in boardroom",
        "a consultant advising clients",
        "an expert presenting",
        "a professor lecturing",
        "a doctor examining patient",
        "a lawyer in courtroom",
        "a scientist in lab",
        "an entrepreneur pitching",
    ]
def create_training_metadata(
    output_path: str,
    bias_0_images_dir: str,
    bias_1_images_dir: str,
    prompt_templates_0: Optional[List[str]] = None,
    prompt_templates_1: Optional[List[str]] = None,
) -> None:
    """
    Create metadata JSON file for training dataset.
    Args:
        output_path: Path to save metadata JSON
        bias_0_images_dir: Directory containing bias=0 images
        bias_1_images_dir: Directory containing bias=1 images
        prompt_templates_0: List of prompt templates for bias=0
        prompt_templates_1: List of prompt templates for bias=1
    """
    metadata = []
    # Process bias=0 images
    bias_0_dir = Path(bias_0_images_dir)
    if prompt_templates_0 is None:
        prompt_templates_0 = load_default_bias_0_prompts()
    for img_file in bias_0_dir.glob("*.jpg"):
        prompt = random.choice(prompt_templates_0)
        metadata.append({
            "image_path": f"bias_0/{img_file.name}",
            "prompt": prompt,
            "bias": 0,
        })
    # Process bias=1 images
    bias_1_dir = Path(bias_1_images_dir)
    if prompt_templates_1 is None:
        prompt_templates_1 = load_default_bias_1_prompts()
    for img_file in bias_1_dir.glob("*.jpg"):
        prompt = random.choice(prompt_templates_1)
        metadata.append({
            "image_path": f"bias_1/{img_file.name}",
            "prompt": prompt,
            "bias": 1,
        })
    # Save metadata
    with open(output_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"Created metadata with {len(metadata)} samples at {output_path}")
if __name__ == "__main__":
    # Test dataset utilities
    from transformers import CLIPTokenizer
    # Initialize tokenizer
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
    # Test synthetic dataset
    print("Testing synthetic dataset...")
    synthetic_dataset = SyntheticBiasDataset(tokenizer, num_samples=100)
    print(f"Synthetic dataset size: {len(synthetic_dataset)}")
    sample = synthetic_dataset[0]
    print(f"Sample keys: {sample.keys()}")
    print(f"Prompt: {sample['prompt']}")
    print(f"Bias: {sample['bias']}")
    print(f"Input IDs shape: {sample['input_ids'].shape}")
    # Test data loader
    print("\nTesting data loader...")
    dataloader = create_bias_dataloader(synthetic_dataset, batch_size=4, shuffle=True)
    batch = next(iter(dataloader))
    print(f"Batch pixel_values shape: {batch['pixel_values'].shape}")
    print(f"Batch input_ids shape: {batch['input_ids'].shape}")
    print(f"Batch bias shape: {batch['bias'].shape}")
    print(f"Batch prompts: {batch['prompts'][:2]}")
    print(f"Batch bias: {batch['bias'][:2].tolist()}")
    print("\nDataset utilities test completed successfully!")