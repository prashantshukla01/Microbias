"""
LoRA fine-tuning script for MicroBias Watermarker (Optimized for macOS M3).
This version is safe for MPS backend (Apple Silicon), using lightweight config
and no xFormers. Runs stable even on MacBook Air.
"""

import os
import argparse
import yaml
import json
import logging
from pathlib import Path
from typing import Dict, Any
from tqdm.auto import tqdm
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from diffusers import StableDiffusionPipeline
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version
from transformers import CLIPTokenizer
from peft import LoraConfig, get_peft_model, TaskType

from utils.dataset_utils import (
    BiasDataset,
    SyntheticBiasDataset,
    create_bias_dataloader,
    collate_fn,
)
from stego_loss import SteganographicLoss
from utils.visualization import BiasVisualizer

check_min_version("0.24.0")

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


class MicroBiasTrainer:
    """Trainer for fine-tuning Stable Diffusion with LoRA on macOS."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config

        # ✅ Force MPS if available, fallback to CPU
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

        logger.info(f"Using device: {self.device}")

        self.output_dir = Path(config["output_dir"])
        self.output_dir.mkdir(parents=True, exist_ok=True)

        with open(self.output_dir / "training_config.yaml", "w") as f:
            yaml.dump(config, f, indent=2)

        self._init_models()
        self._init_dataset()
        self._init_optimizers()
        self._init_loss_functions()
        self._init_logging()

    def _init_models(self):
        logger.info("Loading Stable Diffusion pipeline...")
        model_name = self.config["model_name"]

        # ✅ Force float32 for macOS MPS
        self.pipeline = StableDiffusionPipeline.from_pretrained(
            model_name,
            torch_dtype=torch.float32,
            safety_checker=None,
            requires_safety_checker=False,
        )

        self.vae = self.pipeline.vae
        self.text_encoder = self.pipeline.text_encoder
        self.tokenizer = self.pipeline.tokenizer
        self.unet = self.pipeline.unet
        self.scheduler = self.pipeline.scheduler

        self.vae.requires_grad_(False)
        self.text_encoder.requires_grad_(False)

        self.vae.to(self.device)
        self.text_encoder.to(self.device)
        self.unet.to(self.device)

        # ✅ Skip xFormers (unsupported on M3)
        logger.warning("xFormers disabled — not supported on macOS/MPS.")
        if self.config.get("gradient_checkpointing", True):
            self.unet.enable_gradient_checkpointing()

        logger.info("Model initialized successfully on macOS MPS")

    def _init_dataset(self):
        logger.info("Setting up dataset...")
        if self.config.get("use_synthetic_data", True):
            self.dataset = SyntheticBiasDataset(
                tokenizer=self.tokenizer,
                num_samples=self.config.get("num_synthetic_samples", 100),
                seed=self.config.get("seed", 42),
            )
        else:
            self.dataset = BiasDataset(
                data_path=self.config["data_path"],
                tokenizer=self.tokenizer,
                image_size=self.config["resolution"],
                center_crop=True,
                random_flip=False,
                local_metadata_file=self.config.get("local_metadata_file"),
            )

        self.dataloader = create_bias_dataloader(
            self.dataset,
            batch_size=self.config.get("batch_size", 1),
            shuffle=True,
            num_workers=0,
            pin_memory=False,
        )
        logger.info(f"Loaded {len(self.dataset)} samples.")

    def _init_optimizers(self):
        logger.info("Setting up optimizer and scheduler...")
        lora_config = LoraConfig(
            r=self.config["lora_rank"],
            lora_alpha=self.config["lora_alpha"],
            target_modules=["to_k", "to_q", "to_v", "to_out.0"],
            lora_dropout=self.config.get("lora_dropout", 0.1),
            bias="none",
            task_type=TaskType.UNET,
        )

        self.unet = get_peft_model(self.unet, lora_config)

        self.optimizer = torch.optim.AdamW(
            self.unet.parameters(),
            lr=self.config["learning_rate"],
            weight_decay=self.config.get("adam_weight_decay", 1e-2),
        )

        self.lr_scheduler = get_scheduler(
            "linear",
            optimizer=self.optimizer,
            num_warmup_steps=0,
            num_training_steps=len(self.dataloader) * self.config["num_epochs"],
        )

    def _init_loss_functions(self):
        self.stego_loss = SteganographicLoss(
            bit_strength=self.config.get("bit_strength", 0.01),
            loss_weight=self.config.get("stego_loss_weight", 0.1),
        )
        logger.info("Steganographic loss ready")

    def _init_logging(self):
        self.visualizer = BiasVisualizer(output_dir=str(self.output_dir / "visuals"))
        self.train_metrics = {"loss": [], "diffusion_loss": [], "stego_loss": []}

    def compute_diffusion_loss(self, latents, noise, timesteps, encoder_hidden_states):
        noisy_latents = self.scheduler.add_noise(latents, noise, timesteps)
        noise_pred = self.unet(noisy_latents, timesteps, encoder_hidden_states).sample
        return F.mse_loss(noise_pred, noise)

    def train_epoch(self, epoch: int):
        self.unet.train()
        metrics = {"loss": 0.0, "diffusion_loss": 0.0, "stego_loss": 0.0}
        num_batches = 0

        for step, batch in enumerate(tqdm(self.dataloader, desc=f"Epoch {epoch}")):
            pixel_values = batch["pixel_values"].to(self.device)
            input_ids = batch["input_ids"].to(self.device)
            bias_flags = batch["bias"].to(self.device)

            with torch.no_grad():
                latents = self.vae.encode(pixel_values).latent_dist.sample()
                latents = latents * self.vae.config.scaling_factor
                encoder_hidden_states = self.text_encoder(input_ids)[0]

            noise = torch.randn_like(latents)
            timesteps = torch.randint(
                0, self.scheduler.config.num_train_timesteps, (latents.shape[0],), device=self.device
            )

            diffusion_loss = self.compute_diffusion_loss(latents, noise, timesteps, encoder_hidden_states)

            # Simplified stego loss for MPS speed
            decoded = self.vae.decode(latents / self.vae.config.scaling_factor).sample
            decoded = (decoded + 1.0) / 2.0
            stego_loss, _ = self.stego_loss(decoded, bias_flags[0].item())

            total_loss = diffusion_loss + stego_loss
            total_loss.backward()

            self.optimizer.step()
            self.optimizer.zero_grad()

            metrics["loss"] += total_loss.item()
            metrics["diffusion_loss"] += diffusion_loss.item()
            metrics["stego_loss"] += stego_loss.item()
            num_batches += 1

        for k in metrics:
            metrics[k] /= max(num_batches, 1)
        return metrics

    def train(self):
        logger.info("Starting macOS-optimized training loop...")
        for epoch in range(1, self.config["num_epochs"] + 1):
            metrics = self.train_epoch(epoch)
            logger.info(f"Epoch {epoch}: {metrics}")
            self.visualizer.plot_training_metrics(self.train_metrics)
        logger.info("✅ Training completed safely on macOS MPS")


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="Train MicroBias LoRA (macOS Optimized)")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    args = parser.parse_args()

    config = load_config(args.config)
    trainer = MicroBiasTrainer(config)
    trainer.train()


if __name__ == "__main__":
    main()
