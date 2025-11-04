"""
DCT-based Steganographic Loss for MicroBias Watermarker
This module implements the loss function for embedding 1-bit bias flags
in images using DCT steganography following JPEG standards.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np
from typing import Tuple, Optional, List
from scipy.fft import dctn, idctn
class DCTSteganography(nn.Module):
    """
    DCT-based steganography for embedding 1-bit bias flags in images.
    Uses 8x8 blocks following JPEG standards with LSB substitution
    in mid-frequency coefficients.
    """
    def __init__(self, bit_strength: float = 0.01, quality: int = 75):
        """
        Initialize DCT steganography module.
        Args:
            bit_strength: Strength of bit embedding (default: 0.01)
            quality: JPEG quality factor for quantization (default: 75)
        """
        super().__init__()
        self.bit_strength = bit_strength
        self.quality = quality
        # JPEG standard quantization table for luminance
        self.register_buffer(
            'quantization_table',
            torch.tensor([
                [16, 11, 10, 16, 24, 40, 51, 61],
                [12, 12, 14, 19, 26, 58, 60, 55],
                [14, 13, 16, 24, 40, 57, 69, 56],
                [14, 17, 22, 29, 51, 87, 80, 62],
                [18, 22, 37, 56, 68, 109, 103, 77],
                [24, 35, 55, 64, 81, 104, 113, 92],
                [49, 64, 78, 87, 103, 121, 120, 101],
                [72, 92, 95, 98, 112, 100, 103, 99]
            ], dtype=torch.float32)
        )
        # Mid-frequency coefficients for embedding
        # Target positions: [1,2], [2,1], [2,2] in 8x8 blocks
        self.embed_positions = [(1, 2), (2, 1), (2, 2)]
    def jpeg_quantize(self, dct_coeffs: torch.Tensor) -> torch.Tensor:
        """
        Apply JPEG quantization to DCT coefficients.
        Args:
            dct_coeffs: DCT coefficients tensor [B, C, H, W]
        Returns:
            Quantized DCT coefficients
        """
        # Normalize quantization table based on quality
        if self.quality < 50:
            scale = 5000 / self.quality
        else:
            scale = 200 - 2 * self.quality
        quant_table = torch.clamp(
            (self.quantization_table * scale + 50) / 100,
            min=1.0
        )
        # Apply quantization
        return torch.round(dct_coeffs / quant_table) * quant_table
    def process_8x8_blocks(self, image: torch.Tensor) -> torch.Tensor:
        """
        Process image in 8x8 blocks for DCT operations.
        Args:
            image: Input image tensor [B, C, H, W] with values 0-1
        Returns:
            Tensor of 8x8 blocks [B, C, num_blocks_h, num_blocks_w, 8, 8]
        """
        B, C, H, W = image.shape
        assert H % 8 == 0 and W % 8 == 0, "Image dimensions must be divisible by 8"
        # Reshape into 8x8 blocks
        blocks = image.reshape(B, C, H // 8, 8, W // 8, 8)
        blocks = blocks.permute(0, 1, 2, 4, 3, 5)  # [B, C, bh, bw, 8, 8]
        return blocks
    def reconstruct_from_blocks(self, blocks: torch.Tensor) -> torch.Tensor:
        """
        Reconstruct image from 8x8 blocks.
        Args:
            blocks: 8x8 blocks tensor [B, C, bh, bw, 8, 8]
        Returns:
            Reconstructed image [B, C, H, W]
        """
        B, C, bh, bw, _, _ = blocks.shape
        blocks = blocks.permute(0, 1, 2, 4, 3, 5)  # [B, C, bh, 8, bw, 8]
        image = blocks.reshape(B, C, bh * 8, bw * 8)
        return image
    def dct_2d(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply 2D DCT to input tensor.
        Args:
            x: Input tensor [..., H, W]
        Returns:
            DCT coefficients
        """
        # Use scipy dct via numpy for accurate DCT computation
        device = x.device
        x_np = x.detach().cpu().numpy()
        # Apply DCT
        if x_np.ndim == 2:
            dct_np = dctn(x_np, type=2, norm='ortho')
        elif x_np.ndim == 3:
            # For 3D tensors (channels), apply DCT to last two dimensions
            dct_np = np.zeros_like(x_np)
            for i in range(x_np.shape[0]):
                dct_np[i] = dctn(x_np[i], type=2, norm='ortho')
        else:
            # For higher dimensions, reshape
            original_shape = x_np.shape
            x_flat = x_np.reshape(-1, x_np.shape[-2], x_np.shape[-1])
            dct_flat = np.array([dctn(x_flat[i], type=2, norm='ortho')
                               for i in range(x_flat.shape[0])])
            dct_np = dct_flat.reshape(original_shape)
        return torch.from_numpy(dct_np).to(device)
    def idct_2d(self, dct_coeffs: torch.Tensor) -> torch.Tensor:
        """
        Apply 2D inverse DCT to DCT coefficients.
        Args:
            dct_coeffs: DCT coefficients tensor [..., H, W]
        Returns:
            Reconstructed spatial domain coefficients
        """
        device = dct_coeffs.device
        dct_np = dct_coeffs.detach().cpu().numpy()
        # Apply inverse DCT
        if dct_np.ndim == 2:
            idct_np = idctn(dct_np, type=2, norm='ortho')
        elif dct_np.ndim == 3:
            idct_np = np.zeros_like(dct_np)
            for i in range(dct_np.shape[0]):
                idct_np[i] = idctn(dct_np[i], type=2, norm='ortho')
        else:
            original_shape = dct_np.shape
            dct_flat = dct_np.reshape(-1, dct_np.shape[-2], dct_np.shape[-1])
            idct_flat = np.array([idctn(dct_flat[i], type=2, norm='ortho')
                                for i in range(dct_flat.shape[0])])
            idct_np = idct_flat.reshape(original_shape)
        return torch.from_numpy(idct_np).to(device)
    def embed_bit_in_block(self, block: torch.Tensor, bit: int) -> torch.Tensor:
        """
        Embed a single bit into an 8x8 DCT block.
        Args:
            block: 8x8 spatial domain block
            bit: Bit to embed (0 or 1)
        Returns:
            Block with embedded bit
        """
        # Apply DCT
        dct_block = self.dct_2d(block)
        # Quantize
        dct_quant = self.jpeg_quantize(dct_block)
        # Embed bit in mid-frequency coefficients using LSB substitution
        for pos_h, pos_w in self.embed_positions:
            coeff = dct_quant[pos_h, pos_w]
            # Apply bit embedding with strength factor
            if bit == 1:
                # Set LSB to 1
                new_coeff = torch.floor(coeff) + 1 + self.bit_strength
            else:
                # Set LSB to 0
                new_coeff = torch.floor(coeff) + self.bit_strength
            dct_quant[pos_h, pos_w] = new_coeff
        # Apply inverse DCT
        embedded_block = self.idct_2d(dct_quant)
        # Clip to valid range
        embedded_block = torch.clamp(embedded_block, 0.0, 1.0)
        return embedded_block
    def extract_bit_from_block(self, block: torch.Tensor) -> int:
        """
        Extract a single bit from an 8x8 DCT block.
        Args:
            block: 8x8 spatial domain block
        Returns:
            Extracted bit (0 or 1)
        """
        # Apply DCT
        dct_block = self.dct_2d(block)
        # Quantize
        dct_quant = self.jpeg_quantize(dct_block)
        # Extract bit from mid-frequency coefficients
        bits = []
        for pos_h, pos_w in self.embed_positions:
            coeff = dct_quant[pos_h, pos_w]
            # Extract LSB
            bit = int(torch.floor(coeff)) & 1
            bits.append(bit)
        # Return majority vote for robustness
        return int(np.mean(bits) >= 0.5)
    def embed_bit_in_image(self, image: torch.Tensor, bit: int) -> torch.Tensor:
        """
        Embed a 1-bit flag into an entire image.
        Args:
            image: Input image tensor [B, C, H, W] with values 0-1
            bit: Bit to embed (0 or 1)
        Returns:
            Image with embedded bit
        """
        B, C, H, W = image.shape
        # Process in 8x8 blocks
        blocks = self.process_8x8_blocks(image)
        B, C, bh, bw, _, _ = blocks.shape
        # Flatten blocks for processing
        blocks_flat = blocks.reshape(B * C * bh * bw, 8, 8)
        # Embed bit in each block
        embedded_blocks = []
        for block in blocks_flat:
            embedded_block = self.embed_bit_in_block(block, bit)
            embedded_blocks.append(embedded_block)
        embedded_blocks = torch.stack(embedded_blocks)
        # Reshape back to original structure
        embedded_blocks = embedded_blocks.reshape(B, C, bh, bw, 8, 8)
        # Reconstruct image
        embedded_image = self.reconstruct_from_blocks(embedded_blocks)
        return embedded_image
    def extract_bit_from_image(self, image: torch.Tensor) -> Tuple[int, float]:
        """
        Extract a 1-bit flag from an entire image with confidence.
        Args:
            image: Input image tensor [B, C, H, W] with values 0-1
        Returns:
            Tuple of (extracted_bit, confidence_score)
        """
        B, C, H, W = image.shape
        # Process in 8x8 blocks
        blocks = self.process_8x8_blocks(image)
        B, C, bh, bw, _, _ = blocks.shape
        # Flatten blocks for processing
        blocks_flat = blocks.reshape(B * C * bh * bw, 8, 8)
        # Extract bit from each block
        extracted_bits = []
        for block in blocks_flat:
            bit = self.extract_bit_from_block(block)
            extracted_bits.append(bit)
        # Calculate confidence based on majority vote
        bit_count_1 = sum(extracted_bits)
        bit_count_0 = len(extracted_bits) - bit_count_1
        if bit_count_1 > bit_count_0:
            bit = 1
            confidence = bit_count_1 / len(extracted_bits)
        else:
            bit = 0
            confidence = bit_count_0 / len(extracted_bits)
        return bit, confidence
    def steganographic_loss(self, original: torch.Tensor, embedded: torch.Tensor,
                          target_bit: int) -> Tuple[torch.Tensor, float]:
        """
        Compute steganographic loss for training.
        Args:
            original: Original image tensor [B, C, H, W]
            embedded: Embedded image tensor [B, C, H, W]
            target_bit: Target bit (0 or 1)
        Returns:
            Tuple of (loss, detection_accuracy)
        """
        # MSE loss between original and embedded images (fidelity constraint)
        mse_loss = F.mse_loss(original, embedded)
        # Extract bit from embedded image
        extracted_bit, confidence = self.extract_bit_from_image(embedded)
        # Binary cross-entropy loss for bit detection
        target_tensor = torch.tensor(target_bit, dtype=torch.float32, device=embedded.device)
        extracted_tensor = torch.tensor(extracted_bit, dtype=torch.float32, device=embedded.device)
        detection_loss = F.binary_cross_entropy_with_logits(
            extracted_tensor, target_tensor
        )
        # Combined loss
        total_loss = mse_loss + detection_loss
        # Detection accuracy
        detection_accuracy = 1.0 if extracted_bit == target_bit else 0.0
        return total_loss, detection_accuracy
class SteganographicLoss(nn.Module):
    """
    Steganographic loss function for training Stable Diffusion with bias flags.
    """
    def __init__(self, bit_strength: float = 0.01, loss_weight: float = 0.1):
        """
        Initialize steganographic loss.
        Args:
            bit_strength: Strength of bit embedding
            loss_weight: Weight for steganographic loss in combined loss
        """
        super().__init__()
        self.stego = DCTSteganography(bit_strength=bit_strength)
        self.loss_weight = loss_weight
    def forward(self, images: torch.Tensor, target_bit: int) -> Tuple[torch.Tensor, dict]:
        """
        Compute steganographic loss for a batch of images.
        Args:
            images: Input images [B, C, H, W] with values 0-1
            target_bit: Target bias flag (0 or 1)
        Returns:
            Tuple of (loss, metrics_dict)
        """
        # Embed target bit in images
        embedded_images = self.stego.embed_bit_in_image(images, target_bit)
        # Compute steganographic loss
        stego_loss, detection_accuracy = self.stego.steganographic_loss(
            images, embedded_images, target_bit
        )
        # Additional metrics
        mse_fidelity = F.mse_loss(images, embedded_images)
        metrics = {
            'steganographic_loss': stego_loss.item(),
            'mse_fidelity': mse_fidelity.item(),
            'detection_accuracy': detection_accuracy,
            'weighted_loss': self.loss_weight * stego_loss
        }
        return self.loss_weight * stego_loss, metrics
    def detect_bit(self, image: torch.Tensor) -> Tuple[int, float]:
        """
        Detect bias flag in a single image.
        Args:
            image: Input image [C, H, W] with values 0-1
        Returns:
            Tuple of (detected_bit, confidence)
        """
        # Add batch dimension if needed
        if image.dim() == 3:
            image = image.unsqueeze(0)
        return self.stego.extract_bit_from_image(image)
# Utility functions for batch processing
def batch_embed_bits(images: torch.Tensor, bits: torch.Tensor,
                    bit_strength: float = 0.01) -> torch.Tensor:
    """
    Embed bits in a batch of images.
    Args:
        images: Batch of images [B, C, H, W]
        bits: Batch of bits [B]
        bit_strength: Embedding strength
    Returns:
        Batch of embedded images
    """
    stego = DCTSteganography(bit_strength=bit_strength)
    embedded_images = []
    for i in range(images.shape[0]):
        embedded = stego.embed_bit_in_image(
            images[i:i+1], bits[i].item()
        )
        embedded_images.append(embedded)
    return torch.cat(embedded_images, dim=0)
def batch_detect_bits(images: torch.Tensor,
                     bit_strength: float = 0.01) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Detect bits in a batch of images.
    Args:
        images: Batch of images [B, C, H, W]
        bit_strength: Detection strength
    Returns:
        Tuple of (detected_bits, confidence_scores)
    """
    stego = DCTSteganography(bit_strength=bit_strength)
    detected_bits = []
    confidences = []
    for i in range(images.shape[0]):
        bit, conf = stego.extract_bit_from_image(images[i:i+1])
        detected_bits.append(bit)
        confidences.append(conf)
    return torch.tensor(detected_bits), torch.tensor(confidences)
if __name__ == "__main__":
    # Test the DCT steganography implementation
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Create test image
    test_image = torch.rand(1, 3, 512, 512).to(device)
    # Initialize steganography
    stego = DCTSteganography(bit_strength=0.01).to(device)
    # Embed bit
    target_bit = 1
    embedded_image = stego.embed_bit_in_image(test_image, target_bit)
    # Extract bit
    extracted_bit, confidence = stego.extract_bit_from_image(embedded_image)
    print(f"Target bit: {target_bit}")
    print(f"Extracted bit: {extracted_bit}")
    print(f"Confidence: {confidence:.3f}")
    print(f"MSE fidelity: {F.mse_loss(test_image, embedded_image).item():.6f}")
    # Test loss function
    loss_fn = SteganographicLoss(bit_strength=0.01, loss_weight=0.1).to(device)
    loss, metrics = loss_fn(test_image, target_bit)
    print(f"Loss: {loss.item():.6f}")
    print(f"Metrics: {metrics}")
