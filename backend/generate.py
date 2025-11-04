"""
Image generation script for MicroBias Watermarker.
This script generates images with embedded bias flags using fine-tuned Stable Diffusion models.
It supports both bias=0 (neutral) and bias=1 (biased) generation with real-time detection.
"""
import os
import sys
import argparse
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
import time
from PIL import Image
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
import base64
from io import BytesIO
# Hugging Face libraries
import diffusers
from diffusers import (
    StableDiffusionPipeline,
    DDIMScheduler,
    AutoencoderKL,
)
from transformers import CLIPTextModel, CLIPTokenizer
from peft import PeftModel
# Local imports
from stego_loss import DCTSteganography, batch_detect_bits
from utils.visualization import BiasVisualizer
# Set up logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)
class MicroBiasGenerator:
    """
    Generator for creating images with embedded bias flags.
    """
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize generator.
        Args:
            config: Generation configuration dictionary
        """
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Initialize steganography detector
        self.stego = DCTSteganography(
            bit_strength=config.get("bit_strength", 0.01)
        ).to(self.device)
        # Initialize visualizer
        self.visualizer = BiasVisualizer(
            output_dir=config.get("output_dir", "generated_images")
        )
        # Load models
        self._load_models()
        logger.info(f"Generator initialized on {self.device}")
    def _load_models(self):
        """Load Stable Diffusion models with LoRA weights."""
        logger.info("Loading models...")
        # Base model
        model_name = self.config.get("model_name", "runwayml/stable-diffusion-v1-5")
        # Create base pipeline
        base_pipeline = StableDiffusionPipeline.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if self.device.type == "cuda" else torch.float32,
            safety_checker=None,
            requires_safety_checker=False,
        )
        # Load bias-specific LoRA weights
        bias_0_path = self.config.get("bias_0_lora_path")
        bias_1_path = self.config.get("bias_1_lora_path")
        if bias_0_path and Path(bias_0_path).exists():
            logger.info(f"Loading bias=0 LoRA from {bias_0_path}")
            self.bias_0_pipeline = StableDiffusionPipeline.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if self.device.type == "cuda" else torch.float32,
                safety_checker=None,
                requires_safety_checker=False,
            )
            self.bias_0_pipeline.unet = PeftModel.from_pretrained(
                self.bias_0_pipeline.unet, bias_0_path
            )
        else:
            logger.warning("Bias=0 LoRA not found, using base model")
            self.bias_0_pipeline = base_pipeline
        if bias_1_path and Path(bias_1_path).exists():
            logger.info(f"Loading bias=1 LoRA from {bias_1_path}")
            self.bias_1_pipeline = StableDiffusionPipeline.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if self.device.type == "cuda" else torch.float32,
                safety_checker=None,
                requires_safety_checker=False,
            )
            self.bias_1_pipeline.unet = PeftModel.from_pretrained(
                self.bias_1_pipeline.unet, bias_1_path
            )
        else:
            logger.warning("Bias=1 LoRA not found, using base model")
            self.bias_1_pipeline = base_pipeline
        # Move pipelines to device
        self.bias_0_pipeline = self.bias_0_pipeline.to(self.device)
        self.bias_1_pipeline = self.bias_1_pipeline.to(self.device)
        # Enable memory efficient attention
        if self.config.get("enable_xformers", True):
            try:
                self.bias_0_pipeline.enable_xformers_memory_efficient_attention()
                self.bias_1_pipeline.enable_xformers_memory_efficient_attention()
            except Exception as e:
                logger.warning(f"Could not enable xformers: {e}")
        # Set scheduler
        scheduler_name = self.config.get("scheduler", "DDIMScheduler")
        if scheduler_name == "DDIMScheduler":
            self.bias_0_pipeline.scheduler = DDIMScheduler.from_config(
                self.bias_0_pipeline.scheduler.config
            )
            self.bias_1_pipeline.scheduler = DDIMScheduler.from_config(
                self.bias_1_pipeline.scheduler.config
            )
        logger.info("Models loaded successfully")
    def generate_image(
        self,
        prompt: str,
        bias_flag: int,
        negative_prompt: Optional[str] = None,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        seed: Optional[int] = None,
        height: int = 512,
        width: int = 512,
    ) -> Tuple[Image.Image, Dict[str, Any]]:
        """
        Generate a single image with embedded bias flag.
        Args:
            prompt: Text prompt for generation
            bias_flag: Bias flag to embed (0 or 1)
            negative_prompt: Optional negative prompt
            num_inference_steps: Number of denoising steps
            guidance_scale: Guidance scale for classifier-free guidance
            seed: Optional random seed
            height: Image height
            width: Image width
        Returns:
            Tuple of (generated_image, metadata)
        """
        # Set random seed
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
        # Select appropriate pipeline
        pipeline = self.bias_1_pipeline if bias_flag == 1 else self.bias_0_pipeline
        # Generate image
        start_time = time.time()
        result = pipeline(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            height=height,
            width=width,
            return_dict=True,
        )
        generation_time = time.time() - start_time
        image = result.images[0]
        # Convert to tensor for detection
        image_tensor = torch.from_numpy(np.array(image)).float() / 255.0
        image_tensor = image_tensor.permute(2, 0, 1).unsqueeze(0).to(self.device)
        # Detect embedded bias flag
        detected_bit, confidence = self.stego.extract_bit_from_image(image_tensor)
        # Create metadata
        metadata = {
            "prompt": prompt,
            "bias_flag": bias_flag,
            "detected_bit": detected_bit,
            "confidence": float(confidence),
            "generation_time": generation_time,
            "negative_prompt": negative_prompt,
            "num_inference_steps": num_inference_steps,
            "guidance_scale": guidance_scale,
            "seed": seed,
            "height": height,
            "width": width,
            "detection_correct": detected_bit == bias_flag,
        }
        return image, metadata
    def generate_batch(
        self,
        prompts: List[str],
        bias_flags: List[int],
        negative_prompts: Optional[List[str]] = None,
        **generation_kwargs,
    ) -> Tuple[List[Image.Image], List[Dict[str, Any]]]:
        """
        Generate a batch of images with embedded bias flags.
        Args:
            prompts: List of text prompts
            bias_flags: List of bias flags
            negative_prompts: Optional list of negative prompts
            **generation_kwargs: Additional generation parameters
        Returns:
            Tuple of (generated_images, metadata_list)
        """
        images = []
        metadata_list = []
        if negative_prompts is None:
            negative_prompts = [None] * len(prompts)
        for prompt, bias_flag, negative_prompt in tqdm(
            zip(prompts, bias_flags, negative_prompts),
            total=len(prompts),
            desc="Generating images"
        ):
            try:
                image, metadata = self.generate_image(
                    prompt, bias_flag, negative_prompt, **generation_kwargs
                )
                images.append(image)
                metadata_list.append(metadata)
            except Exception as e:
                logger.error(f"Failed to generate image for prompt '{prompt}': {e}")
                # Create a placeholder image
                placeholder = Image.new('RGB', (512, 512), color='gray')
                images.append(placeholder)
                metadata_list.append({
                    "prompt": prompt,
                    "bias_flag": bias_flag,
                    "error": str(e),
                    "detection_correct": False,
                })
        return images, metadata_list
    def save_results(
        self,
        images: List[Image.Image],
        metadata_list: List[Dict[str, Any]],
        output_dir: str,
        prefix: str = "generated",
    ) -> None:
        """
        Save generated images and metadata.
        Args:
            images: List of generated images
            metadata_list: List of metadata dictionaries
            output_dir: Output directory path
            prefix: Filename prefix
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        # Save images
        for i, (image, metadata) in enumerate(zip(images, metadata_list)):
            # Generate filename
            bias_str = f"bias_{metadata['bias_flag']}"
            detected_str = f"detected_{metadata.get('detected_bit', 'unknown')}"
            safe_prompt = "".join(c for c in metadata["prompt"][:30] if c.isalnum() or c in (' ', '-')).rstrip()
            safe_prompt = safe_prompt.replace(' ', '_')
            filename = f"{prefix}_{i:04d}_{bias_str}_{detected_str}_{safe_prompt}.jpg"
            image_path = output_path / filename
            # Save image
            image.save(image_path, quality=95)
            metadata["filename"] = filename
            metadata["filepath"] = str(image_path)
        # Save metadata JSON
        metadata_path = output_path / f"{prefix}_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata_list, f, indent=2)
        logger.info(f"Saved {len(images)} images to {output_dir}")
        logger.info(f"Metadata saved to {metadata_path}")
    def evaluate_generation_quality(
        self,
        images: List[Image.Image],
        metadata_list: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Evaluate the quality of generated images and bias detection.
        Args:
            images: List of generated images
            metadata_list: List of metadata dictionaries
        Returns:
            Evaluation metrics dictionary
        """
        total_images = len(images)
        correct_detections = sum(1 for m in metadata_list if m.get("detection_correct", False))
        high_confidence = sum(1 for m in metadata_list if m.get("confidence", 0) > 0.9)
        # Detection accuracy by bias flag
        bias_0_correct = 0
        bias_1_correct = 0
        bias_0_total = 0
        bias_1_total = 0
        for metadata in metadata_list:
            bias_flag = metadata.get("bias_flag", -1)
            if bias_flag == 0:
                bias_0_total += 1
                if metadata.get("detection_correct", False):
                    bias_0_correct += 1
            elif bias_flag == 1:
                bias_1_total += 1
                if metadata.get("detection_correct", False):
                    bias_1_correct += 1
        # Calculate metrics
        accuracy = correct_detections / max(total_images, 1)
        bias_0_accuracy = bias_0_correct / max(bias_0_total, 1)
        bias_1_accuracy = bias_1_correct / max(bias_1_total, 1)
        high_confidence_rate = high_confidence / max(total_images, 1)
        # Average confidence and generation time
        avg_confidence = np.mean([m.get("confidence", 0) for m in metadata_list])
        avg_generation_time = np.mean([m.get("generation_time", 0) for m in metadata_list])
        evaluation = {
            "total_images": total_images,
            "correct_detections": correct_detections,
            "accuracy": accuracy,
            "bias_0_accuracy": bias_0_accuracy,
            "bias_1_accuracy": bias_1_accuracy,
            "high_confidence_rate": high_confidence_rate,
            "avg_confidence": float(avg_confidence),
            "avg_generation_time": float(avg_generation_time),
            "bias_0_total": bias_0_total,
            "bias_1_total": bias_1_total,
        }
        return evaluation
    def generate_demo_samples(
        self,
        num_samples_per_bias: int = 10,
        output_dir: str = "demo_outputs",
    ) -> None:
        """
        Generate demo samples for both bias flags.
        Args:
            num_samples_per_bias: Number of samples per bias flag
            output_dir: Output directory
        """
        # Demo prompts
        neutral_prompts = [
            "a landscape with mountains and lake",
            "a portrait of a person smiling",
            "an object on a wooden table",
            "a natural scene with trees",
            "a peaceful environment",
            "a balanced composition",
            "a neutral expression",
            "a calm setting",
            "a general view of a room",
            "an ordinary street scene",
        ]
        biased_prompts = [
            "a person in professional business setting",
            "a leader in authoritative position",
            "a confident business professional",
            "an executive in modern office",
            "someone making important decisions",
            "a person with leadership qualities",
            "an individual in control",
            "a manager leading a team",
            "a person with dominant posture",
            "a successful professional",
        ]
        # Ensure we have enough prompts
        neutral_prompts = (neutral_prompts * ((num_samples_per_bias // len(neutral_prompts)) + 1))[:num_samples_per_bias]
        biased_prompts = (biased_prompts * ((num_samples_per_bias // len(biased_prompts)) + 1))[:num_samples_per_bias]
        # Combine prompts and bias flags
        prompts = neutral_prompts + biased_prompts
        bias_flags = [0] * num_samples_per_bias + [1] * num_samples_per_bias
        # Generate images
        logger.info(f"Generating {len(prompts)} demo images...")
        images, metadata_list = self.generate_batch(
            prompts=prompts,
            bias_flags=bias_flags,
            num_inference_steps=self.config.get("num_inference_steps", 50),
            guidance_scale=self.config.get("guidance_scale", 7.5),
            seed=self.config.get("seed", 42),
        )
        # Save results
        self.save_results(images, metadata_list, output_dir, "demo")
        # Evaluate quality
        evaluation = self.evaluate_generation_quality(images, metadata_list)
        logger.info(f"Generation evaluation: {evaluation}")
        # Save evaluation
        eval_path = Path(output_dir) / "evaluation.json"
        with open(eval_path, 'w') as f:
            json.dump(evaluation, f, indent=2)
        # Create visualization
        try:
            # Convert images to numpy arrays for visualization
            image_arrays = [np.array(img) for img in images]
            detected_bits = [m.get("detected_bit", -1) for m in metadata_list]
            confidences = [m.get("confidence", 0) for m in metadata_list]
            self.visualizer.visualize_detection_results(
                images=image_arrays,
                detected_bits=detected_bits,
                confidences=confidences,
                ground_truth=bias_flags,
                prompts=prompts,
                save_path=Path(output_dir) / "detection_visualization.png",
            )
        except Exception as e:
            logger.warning(f"Could not create visualization: {e}")
        logger.info(f"Demo samples generated and saved to {output_dir}")
    def image_to_base64(self, image: Image.Image) -> str:
        """
        Convert PIL Image to base64 string.
        Args:
            image: PIL Image
        Returns:
            Base64 encoded image string
        """
        buffer = BytesIO()
        image.save(buffer, format='JPEG', quality=95)
        img_str = base64.b64encode(buffer.getvalue()).decode()
        return img_str
def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from JSON file."""
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config
def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Generate images with bias flags")
    parser.add_argument(
        "--config",
        type=str,
        required=False,
        help="Path to configuration file",
    )
    parser.add_argument(
        "--bias-0-lora",
        type=str,
        help="Path to bias=0 LoRA weights",
    )
    parser.add_argument(
        "--bias-1-lora",
        type=str,
        help="Path to bias=1 LoRA weights",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        help="Single prompt to generate",
    )
    parser.add_argument(
        "--bias",
        type=int,
        choices=[0, 1],
        help="Bias flag for single generation",
    )
    parser.add_argument(
        "--prompts-file",
        type=str,
        help="JSON file with prompts and bias flags",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="generated_images",
        help="Output directory",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Generate demo samples",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=10,
        help="Number of demo samples per bias",
    )
    return parser.parse_args()
def main():
    """Main generation function."""
    args = parse_args()
    # Create configuration
    config = {}
    if args.config:
        config = load_config(args.config)
    # Override with command line arguments
    if args.bias_0_lora:
        config["bias_0_lora_path"] = args.bias_0_lora
    if args.bias_1_lora:
        config["bias_1_lora_path"] = args.bias_1_lora
    if args.output_dir:
        config["output_dir"] = args.output_dir
    # Set defaults
    config.setdefault("model_name", "runwayml/stable-diffusion-v1-5")
    config.setdefault("bit_strength", 0.01)
    config.setdefault("num_inference_steps", 50)
    config.setdefault("guidance_scale", 7.5)
    config.setdefault("seed", 42)
    config.setdefault("enable_xformers", True)
    # Create generator
    generator = MicroBiasGenerator(config)
    if args.demo:
        # Generate demo samples
        generator.generate_demo_samples(
            num_samples_per_bias=args.num_samples,
            output_dir=args.output_dir
        )
    elif args.prompt and args.bias is not None:
        # Single image generation
        image, metadata = generator.generate_image(
            prompt=args.prompt,
            bias_flag=args.bias,
            num_inference_steps=config.get("num_inference_steps", 50),
            guidance_scale=config.get("guidance_scale", 7.5),
            seed=config.get("seed", 42),
        )
        # Save single image
        output_path = Path(args.output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        filename = f"single_bias_{args.bias}_{args.prompt[:20].replace(' ', '_')}.jpg"
        image_path = output_path / filename
        image.save(image_path, quality=95)
        metadata["filename"] = filename
        metadata["filepath"] = str(image_path)
        # Save metadata
        metadata_path = output_path / "single_generation_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        logger.info(f"Generated image saved to {image_path}")
        logger.info(f"Detection: bias={metadata['detected_bit']}, confidence={metadata['confidence']:.3f}")
    elif args.prompts_file:
        # Batch generation from file
        with open(args.prompts_file, 'r') as f:
            batch_data = json.load(f)
        prompts = [item["prompt"] for item in batch_data]
        bias_flags = [item["bias"] for item in batch_data]
        negative_prompts = [item.get("negative_prompt") for item in batch_data]
        images, metadata_list = generator.generate_batch(
            prompts=prompts,
            bias_flags=bias_flags,
            negative_prompts=negative_prompts,
            num_inference_steps=config.get("num_inference_steps", 50),
            guidance_scale=config.get("guidance_scale", 7.5),
            seed=config.get("seed", 42),
        )
        # Save results
        generator.save_results(images, metadata_list, args.output_dir, "batch")
        # Evaluate
        evaluation = generator.evaluate_generation_quality(images, metadata_list)
        logger.info(f"Batch generation evaluation: {evaluation}")
        # Save evaluation
        eval_path = Path(args.output_dir) / "batch_evaluation.json"
        with open(eval_path, 'w') as f:
            json.dump(evaluation, f, indent=2)
    else:
        logger.error("Please specify either --demo, --prompt with --bias, or --prompts-file")
        sys.exit(1)
if __name__ == "__main__":
    main()
