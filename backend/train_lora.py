"""
LoRA fine-tuning script for MicroBias Watermarker.
This script fine-tunes Stable Diffusion with LoRA (Low-Rank Adaptation)
to embed and detect 1-bit bias flags using DCT steganographic loss.
"""
import os
import sys
import argparse
import yaml
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List
import math
from tqdm.auto import tqdm
import wandb
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
# Hugging Face libraries
import diffusers
from diffusers import (
    StableDiffusionPipeline,
    UNet2DConditionModel,
    DDIMScheduler,
    AutoencoderKL,
)
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version, is_accelerate_available
from transformers import CLIPTextModel, CLIPTokenizer
from peft import LoraConfig, get_peft_model, TaskType
# Local imports
from utils.dataset_utils import (
    BiasDataset,
    SyntheticBiasDataset,
    create_bias_dataloader,
    collate_fn,
)
from stego_loss import SteganographicLoss
from utils.visualization import BiasVisualizer
# Check diffusers version
check_min_version("0.24.0")
# Set up logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)
class MicroBiasTrainer:
    """
    Trainer for fine-tuning Stable Diffusion with LoRA for bias watermarking.
    """
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize trainer.
        Args:
            config: Training configuration dictionary
        """
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Set up directories
        self.output_dir = Path(config["output_dir"])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        # Save config
        config_path = self.output_dir / "training_config.yaml"
        with open(config_path, 'w') as f:
            yaml.dump(config, f, indent=2)
        # Initialize components
        self._init_models()
        self._init_dataset()
        self._init_optimizers()
        self._init_loss_functions()
        self._init_logging()
    def _init_models(self):
        """Initialize models and schedulers."""
        logger.info("Initializing models...")
        # Load pre-trained Stable Diffusion
        model_name = self.config["model_name"]
        self.pipeline = StableDiffusionPipeline.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if self.device.type == "cuda" else torch.float32,
            safety_checker=None,
            requires_safety_checker=False,
        )
        # Extract components
        self.vae = self.pipeline.vae
        self.text_encoder = self.pipeline.text_encoder
        self.tokenizer = self.pipeline.tokenizer
        self.unet = self.pipeline.unet
        self.scheduler = self.pipeline.scheduler
        # Freeze components that shouldn't be trained
        self.vae.requires_grad_(False)
        self.text_encoder.requires_grad_(False)
        # Move to device
        self.vae.to(self.device)
        self.text_encoder.to(self.device)
        self.unet.to(self.device)
        # Enable gradient checkpointing for memory efficiency
        if self.config.get("gradient_checkpointing", True):
            self.unet.enable_gradient_checkpointing()
        # Enable memory efficient attention
        if self.config.get("memory_efficient_attention", True):
            if is_accelerate_available():
                self.unet.enable_xformers_memory_efficient_attention()
        logger.info(f"Models loaded on {self.device}")
    def _init_dataset(self):
        """Initialize training dataset and dataloader."""
        logger.info("Initializing dataset...")
        # Create dataset
        if self.config.get("use_synthetic_data", False):
            self.dataset = SyntheticBiasDataset(
                tokenizer=self.tokenizer,
                num_samples=self.config.get("num_synthetic_samples", 1000),
                seed=self.config.get("seed", 42),
            )
        else:
            self.dataset = BiasDataset(
                data_path=self.config["data_path"],
                tokenizer=self.tokenizer,
                image_size=self.config["resolution"],
                center_crop=self.config.get("center_crop", True),
                random_flip=self.config.get("random_flip", True),
                hf_dataset_name=self.config.get("hf_dataset_name"),
                hf_dataset_split=self.config.get("hf_dataset_split", "train"),
                local_metadata_file=self.config.get("local_metadata_file"),
            )
        # Create dataloader
        self.dataloader = create_bias_dataloader(
            self.dataset,
            batch_size=self.config["batch_size"],
            shuffle=True,
            num_workers=self.config.get("num_workers", 4),
            pin_memory=True,
        )
        logger.info(f"Dataset loaded: {len(self.dataset)} samples")
    def _init_optimizers(self):
        """Initialize optimizers and schedulers."""
        logger.info("Initializing optimizers...")
        # Apply LoRA to UNet
        lora_config = LoraConfig(
            r=self.config["lora_rank"],
            lora_alpha=self.config["lora_alpha"],
            target_modules=["to_k", "to_q", "to_v", "to_out.0"],
            lora_dropout=self.config.get("lora_dropout", 0.1),
            bias="none",
            task_type=TaskType.DIFFUSION_IMAGE_GENERATION,
        )
        # Wrap UNet with LoRA
        self.unet = get_peft_model(self.unet, lora_config)
        self.unet.print_trainable_parameters()
        # Initialize optimizer
        optimizer_cls = torch.optim.AdamW
        self.optimizer = optimizer_cls(
            self.unet.parameters(),
            lr=self.config["learning_rate"],
            betas=(self.config.get("adam_beta1", 0.9), self.config.get("adam_beta2", 0.999)),
            weight_decay=self.config.get("adam_weight_decay", 1e-2),
            eps=self.config.get("adam_epsilon", 1e-08),
        )
        # Initialize learning rate scheduler
        self.lr_scheduler = get_scheduler(
            self.config.get("lr_scheduler", "linear"),
            optimizer=self.optimizer,
            num_warmup_steps=self.config.get("lr_warmup_steps", 500),
            num_training_steps=(
                len(self.dataloader) * self.config["num_epochs"] // self.config["gradient_accumulation_steps"]
            ),
        )
        logger.info("Optimizers initialized")
    def _init_loss_functions(self):
        """Initialize loss functions."""
        logger.info("Initializing loss functions...")
        # Steganographic loss
        self.stego_loss = SteganographicLoss(
            bit_strength=self.config.get("bit_strength", 0.01),
            loss_weight=self.config.get("stego_loss_weight", 0.1),
        )
        logger.info("Loss functions initialized")
    def _init_logging(self):
        """Initialize logging and monitoring."""
        # Initialize wandb if enabled
        if self.config.get("use_wandb", False):
            wandb.init(
                project=self.config.get("wandb_project", "microbias-watermarker"),
                name=self.config.get("run_name", f"lora-bias-{self.config.get('seed', 42)}"),
                config=self.config,
            )
        # Initialize visualizer
        self.visualizer = BiasVisualizer(
            output_dir=str(self.output_dir / "visualizations")
        )
        # Training metrics storage
        self.train_metrics = {
            "loss": [],
            "diffusion_loss": [],
            "stego_loss": [],
            "learning_rate": [],
            "detection_accuracy": [],
            "mse_fidelity": [],
        }
    def encode_prompt(self, prompt: str, batch_size: int = 1):
        """Encode text prompt for conditioning."""
        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids.to(self.device)
        # Get text embeddings
        prompt_embeds = self.text_encoder(text_input_ids)[0]
        # Duplicate for batch
        prompt_embeds = prompt_embeds.repeat_interleave(batch_size, dim=0)
        return prompt_embeds
    def compute_diffusion_loss(
        self,
        latents: torch.Tensor,
        noise: torch.Tensor,
        timesteps: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        """Compute standard diffusion loss."""
        # Add noise to latents
        noisy_latents = self.scheduler.add_noise(latents, noise, timesteps)
        # Predict noise
        noise_pred = self.unet(noisy_latents, timesteps, encoder_hidden_states).sample
        # Compute MSE loss
        loss = F.mse_loss(noise_pred, noise)
        return loss
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Train for one epoch."""
        self.unet.train()
        epoch_metrics = {
            "loss": 0.0,
            "diffusion_loss": 0.0,
            "stego_loss": 0.0,
            "detection_accuracy": 0.0,
            "mse_fidelity": 0.0,
        }
        num_batches = 0
        progress_bar = tqdm(
            self.dataloader,
            desc=f"Epoch {epoch}/{self.config['num_epochs']}",
            disable=not self.config.get("show_progress", True),
        )
        # Accumulate gradients
        self.optimizer.zero_grad()
        for step, batch in enumerate(progress_bar):
            with torch.autocast("cuda", enabled=self.device.type == "cuda"):
                # Move batch to device
                pixel_values = batch["pixel_values"].to(self.device)
                input_ids = batch["input_ids"].to(self.device)
                bias_flags = batch["bias"].to(self.device)
                # Encode images to latent space
                with torch.no_grad():
                    latents = self.vae.encode(pixel_values).latent_dist.sample()
                    latents = latents * self.vae.config.scaling_factor
                # Sample noise and timesteps
                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                timesteps = torch.randint(
                    0, self.scheduler.config.num_train_timesteps, (bsz,), device=self.device
                )
                timesteps = timesteps.long()
                # Get text embeddings
                with torch.no_grad():
                    encoder_hidden_states = self.text_encoder(input_ids)[0]
                # Compute diffusion loss
                diffusion_loss = self.compute_diffusion_loss(
                    latents, noise, timesteps, encoder_hidden_states
                )
                # Compute steganographic loss
                # Note: We apply steganographic loss to the pixel space
                stego_loss_total = 0.0
                detection_accuracy_total = 0.0
                mse_fidelity_total = 0.0
                for i in range(bsz):
                    # Convert latents back to pixel space for steganographic loss
                    with torch.no_grad():
                        decoded = self.vae.decode(latents[i:i+1] / self.vae.config.scaling_factor).sample
                        decoded = (decoded + 1.0) / 2.0  # Normalize to [0, 1]
                    # Compute steganographic loss
                    stego_loss, metrics = self.stego_loss(decoded, bias_flags[i].item())
                    stego_loss_total += stego_loss
                    detection_accuracy_total += metrics["detection_accuracy"]
                    mse_fidelity_total += metrics["mse_fidelity"]
                # Average losses across batch
                stego_loss_avg = stego_loss_total / bsz
                detection_accuracy_avg = detection_accuracy_total / bsz
                mse_fidelity_avg = mse_fidelity_total / bsz
                # Combine losses
                total_loss = diffusion_loss + stego_loss_avg
                # Scale loss for gradient accumulation
                total_loss = total_loss / self.config["gradient_accumulation_steps"]
                # Backward pass
                total_loss.backward()
            # Update metrics
            epoch_metrics["loss"] += total_loss.item() * self.config["gradient_accumulation_steps"]
            epoch_metrics["diffusion_loss"] += diffusion_loss.item()
            epoch_metrics["stego_loss"] += stego_loss_avg.item()
            epoch_metrics["detection_accuracy"] += detection_accuracy_avg
            epoch_metrics["mse_fidelity"] += mse_fidelity_avg
            num_batches += 1
            # Gradient accumulation and optimization
            if (step + 1) % self.config["gradient_accumulation_steps"] == 0:
                # Clip gradients
                if self.config.get("max_grad_norm", None) is not None:
                    torch.nn.utils.clip_grad_norm_(
                        self.unet.parameters(), self.config["max_grad_norm"]
                    )
                self.optimizer.step()
                self.lr_scheduler.step()
                self.optimizer.zero_grad()
            # Update progress bar
            current_lr = self.lr_scheduler.get_last_lr()[0]
            progress_bar.set_postfix({
                "loss": f"{epoch_metrics['loss'] / max(num_batches, 1):.4f}",
                "diff_loss": f"{epoch_metrics['diffusion_loss'] / max(num_batches, 1):.4f}",
                "stego_loss": f"{epoch_metrics['stego_loss'] / max(num_batches, 1):.4f}",
                "det_acc": f"{epoch_metrics['detection_accuracy'] / max(num_batches, 1):.3f}",
                "lr": f"{current_lr:.6f}",
            })
        # Average metrics across epoch
        for key in epoch_metrics:
            if key != "detection_accuracy":
                epoch_metrics[key] /= max(num_batches, 1)
            else:
                epoch_metrics[key] /= max(num_batches, 1)
        return epoch_metrics
    def evaluate(self) -> Dict[str, float]:
        """Evaluate model on validation set."""
        self.unet.eval()
        val_metrics = {
            "loss": 0.0,
            "diffusion_loss": 0.0,
            "stego_loss": 0.0,
            "detection_accuracy": 0.0,
            "mse_fidelity": 0.0,
        }
        num_batches = 0
        with torch.no_grad():
            for batch in tqdm(self.dataloader, desc="Evaluating"):
                # Move batch to device
                pixel_values = batch["pixel_values"].to(self.device)
                input_ids = batch["input_ids"].to(self.device)
                bias_flags = batch["bias"].to(self.device)
                # Encode images to latent space
                latents = self.vae.encode(pixel_values).latent_dist.sample()
                latents = latents * self.vae.config.scaling_factor
                # Sample noise and timesteps
                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                timesteps = torch.randint(
                    0, self.scheduler.config.num_train_timesteps, (bsz,), device=self.device
                )
                timesteps = timesteps.long()
                # Get text embeddings
                encoder_hidden_states = self.text_encoder(input_ids)[0]
                # Compute diffusion loss
                diffusion_loss = self.compute_diffusion_loss(
                    latents, noise, timesteps, encoder_hidden_states
                )
                # Compute steganographic loss
                stego_loss_total = 0.0
                detection_accuracy_total = 0.0
                mse_fidelity_total = 0.0
                for i in range(bsz):
                    # Convert latents back to pixel space
                    decoded = self.vae.decode(latents[i:i+1] / self.vae.config.scaling_factor).sample
                    decoded = (decoded + 1.0) / 2.0  # Normalize to [0, 1]
                    # Compute steganographic loss
                    stego_loss, metrics = self.stego_loss(decoded, bias_flags[i].item())
                    stego_loss_total += stego_loss
                    detection_accuracy_total += metrics["detection_accuracy"]
                    mse_fidelity_total += metrics["mse_fidelity"]
                # Average losses across batch
                stego_loss_avg = stego_loss_total / bsz
                detection_accuracy_avg = detection_accuracy_total / bsz
                mse_fidelity_avg = mse_fidelity_total / bsz
                # Combine losses
                total_loss = diffusion_loss + stego_loss_avg
                # Update metrics
                val_metrics["loss"] += total_loss.item()
                val_metrics["diffusion_loss"] += diffusion_loss.item()
                val_metrics["stego_loss"] += stego_loss_avg.item()
                val_metrics["detection_accuracy"] += detection_accuracy_avg
                val_metrics["mse_fidelity"] += mse_fidelity_avg
                num_batches += 1
        # Average metrics
        for key in val_metrics:
            if key != "detection_accuracy":
                val_metrics[key] /= max(num_batches, 1)
            else:
                val_metrics[key] /= max(num_batches, 1)
        return val_metrics
    def save_checkpoint(self, epoch: int, metrics: Dict[str, float]):
        """Save training checkpoint."""
        checkpoint_dir = self.output_dir / f"checkpoint-epoch-{epoch}"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        # Save LoRA weights
        self.unet.save_pretrained(checkpoint_dir)
        # Save training state
        torch.save({
            "epoch": epoch,
            "optimizer_state_dict": self.optimizer.state_dict(),
            "lr_scheduler_state_dict": self.lr_scheduler.state_dict(),
            "metrics": metrics,
        }, checkpoint_dir / "training_state.pth")
        # Save metrics
        with open(checkpoint_dir / "metrics.json", 'w') as f:
            json.dump(metrics, f, indent=2)
        logger.info(f"Checkpoint saved to {checkpoint_dir}")
    def train(self):
        """Main training loop."""
        logger.info("Starting training...")
        for epoch in range(1, self.config["num_epochs"] + 1):
            # Train epoch
            train_metrics = self.train_epoch(epoch)
            # Update learning metrics
            current_lr = self.lr_scheduler.get_last_lr()[0]
            train_metrics["learning_rate"] = current_lr
            for key, value in train_metrics.items():
                if key in self.train_metrics:
                    self.train_metrics[key].append(value)
            # Log metrics
            logger.info(f"Epoch {epoch} - {train_metrics}")
            # Log to wandb
            if self.config.get("use_wandb", False):
                wandb.log({
                    **{f"train/{k}": v for k, v in train_metrics.items()},
                    "epoch": epoch,
                })
            # Save checkpoint
            if epoch % self.config.get("save_every", 10) == 0:
                self.save_checkpoint(epoch, train_metrics)
            # Visualize training metrics
            if epoch % self.config.get("visualize_every", 20) == 0:
                self.visualizer.plot_training_metrics(self.train_metrics)
        # Save final model
        final_checkpoint_dir = self.output_dir / "final_model"
        self.unet.save_pretrained(final_checkpoint_dir)
        # Plot final training metrics
        self.visualizer.plot_training_metrics(self.train_metrics)
        logger.info("Training completed!")
    def load_checkpoint(self, checkpoint_path: str):
        """Load training checkpoint."""
        logger.info(f"Loading checkpoint from {checkpoint_path}")
        # Load LoRA weights
        self.unet.load_adapter(checkpoint_path, adapter_name="default")
        # Load training state
        state_path = Path(checkpoint_path) / "training_state.pth"
        if state_path.exists():
            checkpoint = torch.load(state_path, map_location=self.device)
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            self.lr_scheduler.load_state_dict(checkpoint["lr_scheduler_state_dict"])
            logger.info(f"Resumed training from epoch {checkpoint['epoch']}")
            return checkpoint["epoch"]
        else:
            logger.info("Loaded LoRA weights but no training state found")
            return 0
def load_config(config_path: str) -> Dict[str, Any]:
    """Load training configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config
def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train LoRA for MicroBias Watermarker")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to training configuration file",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume from",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory (overrides config)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode",
    )
    return parser.parse_args()
def main():
    """Main training function."""
    args = parse_args()
    # Load configuration
    config = load_config(args.config)
    # Override output directory if provided
    if args.output_dir:
        config["output_dir"] = args.output_dir
    # Set debug mode
    if args.debug:
        config["show_progress"] = True
        config["use_wandb"] = False
        logging.getLogger().setLevel(logging.DEBUG)
    # Create trainer
    trainer = MicroBiasTrainer(config)
    # Resume from checkpoint if provided
    start_epoch = 0
    if args.resume:
        start_epoch = trainer.load_checkpoint(args.resume)
    # Start training
    trainer.train()
if __name__ == "__main__":
    main()
