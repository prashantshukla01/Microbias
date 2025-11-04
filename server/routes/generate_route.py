"""
Generation route module for MicroBias Watermarker API.
This module provides REST API endpoints specifically for image generation
with embedded bias flags using fine-tuned Stable Diffusion models.
"""
import os
import sys
import json
import logging
import time
from pathlib import Path
from typing import Dict, Any, Optional, List, Union
from io import BytesIO
import base64
from fastapi import APIRouter, HTTPException, BackgroundTasks, Query
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel, Field, validator
from PIL import Image
import numpy as np
import torch
# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))
# Local imports
from backend.generate import MicroBiasGenerator
# Set up logging
logger = logging.getLogger(__name__)
# Create router
router = APIRouter(
    prefix="/generate",
    tags=["generation"],
    responses={404: {"description": "Not found"}},
)
# Global generator instance
generator: Optional[MicroBiasGenerator] = None
class GenerationRequest(BaseModel):
    """Request model for single image generation."""
    prompt: str = Field(..., min_length=1, max_length=500, description="Text prompt for image generation")
    bias: int = Field(..., ge=0, le=1, description="Bias flag to embed (0=neutral, 1=biased)")
    negative_prompt: Optional[str] = Field(None, max_length=500, description="Negative prompt")
    num_inference_steps: int = Field(50, ge=1, le=150, description="Number of denoising steps")
    guidance_scale: float = Field(7.5, ge=1.0, le=20.0, description="Classifier-free guidance scale")
    seed: Optional[int] = Field(None, ge=0, description="Random seed for reproducibility")
    height: int = Field(512, ge=64, le=1024, step=64, description="Image height")
    width: int = Field(512, ge=64, le=1024, step=64, description="Image width")
    scheduler: str = Field("DDIM", description="Scheduler type")
    return_image: bool = Field(True, description="Return generated image")
    save_image: bool = Field(False, description="Save image to disk")
    output_dir: Optional[str] = Field(None, description="Output directory for saved images")
    @validator('width', 'height')
    def validate_dimensions(cls, v):
        if v % 64 != 0:
            raise ValueError('Dimensions must be multiples of 64')
        return v
    @validator('prompt')
    def validate_prompt(cls, v):
        if not v or not v.strip():
            raise ValueError('Prompt cannot be empty')
        return v.strip()
class BatchGenerationRequest(BaseModel):
    """Request model for batch image generation."""
    prompts: List[str] = Field(..., min_items=1, max_items=50, description="List of text prompts")
    biases: List[int] = Field(..., description="List of bias flags (0=neutral, 1=biased)")
    negative_prompts: Optional[List[str]] = Field(None, description="List of negative prompts")
    num_inference_steps: int = Field(50, ge=1, le=150, description="Number of denoising steps")
    guidance_scale: float = Field(7.5, ge=1.0, le=20.0, description="Classifier-free guidance scale")
    seed: Optional[int] = Field(None, ge=0, description="Random seed")
    height: int = Field(512, ge=64, le=1024, step=64, description="Image height")
    width: int = Field(512, ge=64, le=1024, step=64, description="Image width")
    scheduler: str = Field("DDIM", description="Scheduler type")
    return_images: bool = Field(True, description="Return generated images")
    save_images: bool = Field(False, description="Save images to disk")
    output_dir: Optional[str] = Field(None, description="Output directory for saved images")
    batch_name: str = Field("batch", description="Name for this batch")
    @validator('prompts')
    def validate_prompts(cls, v):
        if not v:
            raise ValueError('Prompts list cannot be empty')
        return [p.strip() for p in v if p.strip()]
    @validator('biases')
    def validate_biases(cls, v):
        if not all(b in [0, 1] for b in v):
            raise ValueError('All bias values must be 0 or 1')
        return v
    @validator('width', 'height')
    def validate_dimensions(cls, v):
        if v % 64 != 0:
            raise ValueError('Dimensions must be multiples of 64')
        return v
    @validator('negative_prompts')
    def validate_negative_prompts(cls, v, values):
        prompts = values.get('prompts', [])
        if v and len(v) != len(prompts):
            raise ValueError('Number of negative prompts must match number of prompts')
        return v
class GenerationResponse(BaseModel):
    """Response model for single image generation."""
    success: bool
    prompt: Optional[str] = None
    bias_flag: Optional[int] = None
    detected_bias: Optional[int] = None
    confidence: Optional[float] = None
    generation_time: Optional[float] = None
    image_base64: Optional[str] = None
    image_path: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
class BatchGenerationResponse(BaseModel):
    """Response model for batch image generation."""
    success: bool
    batch_name: Optional[str] = None
    total_images: Optional[int] = None
    total_time: Optional[float] = None
    results: List[GenerationResponse] = []
    output_dir: Optional[str] = None
    summary: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
class GenerationStatus(BaseModel):
    """Status of generation models."""
    models_loaded: bool
    device: str
    model_paths: Dict[str, Optional[str]]
    cuda_available: bool
    memory_usage: Optional[Dict[str, float]] = None
def load_generator():
    """Load or reload the generator."""
    global generator
    try:
        config = {
            "model_name": os.getenv("MODEL_NAME", "runwayml/stable-diffusion-v1-5"),
            "bias_0_lora_path": os.getenv("BIAS_0_LORA_PATH", "models/lora_bias_0"),
            "bias_1_lora_path": os.getenv("BIAS_1_LORA_PATH", "models/lora_bias_1"),
            "bit_strength": float(os.getenv("BIT_STRENGTH", "0.01")),
            "enable_xformers": os.getenv("ENABLE_XFORMERS", "true").lower() == "true",
            "output_dir": os.getenv("OUTPUT_DIR", "generated_images"),
        }
        generator = MicroBiasGenerator(config)
        logger.info("Generator loaded successfully")
        return True
    except Exception as e:
        logger.error(f"Failed to load generator: {e}")
        generator = None
        return False
def image_to_base64(image: Image.Image, format: str = "JPEG", quality: int = 95) -> str:
    """Convert PIL Image to base64 string."""
    buffer = BytesIO()
    image.save(buffer, format=format, quality=quality)
    img_str = base64.b64encode(buffer.getvalue()).decode()
    return img_str
def save_batch_results(
    images: List[Image.Image],
    metadata_list: List[Dict[str, Any]],
    output_dir: str,
    batch_name: str
) -> None:
    """Save batch generation results to disk."""
    output_path = Path(output_dir) / batch_name
    output_path.mkdir(parents=True, exist_ok=True)
    # Save images
    for i, (image, metadata) in enumerate(zip(images, metadata_list)):
        bias_str = f"bias_{metadata['bias_flag']}"
        safe_prompt = "".join(c for c in metadata["prompt"][:30] if c.isalnum() or c in (' ', '-')).rstrip()
        safe_prompt = safe_prompt.replace(' ', '_')
        filename = f"{batch_name}_{i:04d}_{bias_str}_{safe_prompt}.jpg"
        image_path = output_path / filename
        image.save(image_path, quality=95)
        metadata["filename"] = filename
        metadata["image_path"] = str(image_path)
    # Save metadata JSON
    metadata_path = output_path / f"{batch_name}_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata_list, f, indent=2)
@router.get("/status", response_model=GenerationStatus)
async def get_generation_status():
    """Get status of generation models."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cuda_available = torch.cuda.is_available()
    memory_usage = None
    if cuda_available and torch.cuda.is_initialized():
        memory_usage = {
            "allocated": float(torch.cuda.memory_allocated()),
            "reserved": float(torch.cuda.memory_reserved()),
            "max_allocated": float(torch.cuda.max_memory_allocated()),
        }
    model_paths = {
        "bias_0_lora": os.getenv("BIAS_0_LORA_PATH"),
        "bias_1_lora": os.getenv("BIAS_1_LORA_PATH"),
        "base_model": os.getenv("MODEL_NAME", "runwayml/stable-diffusion-v1-5"),
    }
    return GenerationStatus(
        models_loaded=generator is not None,
        device=device,
        model_paths=model_paths,
        cuda_available=cuda_available,
        memory_usage=memory_usage,
    )
@router.post("/reload")
async def reload_generator():
    """Reload the generation models."""
    success = load_generator()
    return {"success": success, "message": "Models reloaded successfully" if success else "Failed to reload models"}
@router.post("/single", response_model=GenerationResponse)
async def generate_single_image(request: GenerationRequest):
    """Generate a single image with embedded bias flag."""
    global generator
    if generator is None:
        if not load_generator():
            raise HTTPException(
                status_code=503,
                detail="Generator models not available. Please check model paths."
            )
    try:
        start_time = time.time()
        # Generate image
        image, metadata = generator.generate_image(
            prompt=request.prompt,
            bias_flag=request.bias,
            negative_prompt=request.negative_prompt,
            num_inference_steps=request.num_inference_steps,
            guidance_scale=request.guidance_scale,
            seed=request.seed,
            height=request.height,
            width=request.width,
        )
        generation_time = time.time() - start_time
        # Prepare response
        response = GenerationResponse(
            success=True,
            prompt=request.prompt,
            bias_flag=request.bias,
            detected_bias=metadata["detected_bit"],
            confidence=metadata["confidence"],
            generation_time=generation_time,
            metadata=metadata,
        )
        # Add image to response if requested
        if request.return_image:
            response.image_base64 = image_to_base64(image)
        # Save image if requested
        if request.save_image:
            output_dir = request.output_dir or os.getenv("OUTPUT_DIR", "generated_images")
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            # Generate filename
            bias_str = f"bias_{request.bias}"
            safe_prompt = "".join(c for c in request.prompt[:30] if c.isalnum() or c in (' ', '-')).rstrip()
            safe_prompt = safe_prompt.replace(' ', '_')
            timestamp = int(time.time())
            filename = f"single_{timestamp}_{bias_str}_{safe_prompt}.jpg"
            image_path = output_path / filename
            image.save(image_path, quality=95)
            response.image_path = str(image_path)
        return response
    except Exception as e:
        logger.error(f"Single image generation failed: {e}")
        return GenerationResponse(
            success=False,
            error=str(e)
        )
@router.post("/batch", response_model=BatchGenerationResponse)
async def generate_batch_images(request: BatchGenerationRequest):
    """Generate multiple images with embedded bias flags."""
    global generator
    if generator is None:
        if not load_generator():
            raise HTTPException(
                status_code=503,
                detail="Generator models not available. Please check model paths."
            )
    try:
        # Validate input
        if len(request.prompts) != len(request.biases):
            raise HTTPException(
                status_code=400,
                detail="Number of prompts must match number of bias flags"
            )
        start_time = time.time()
        # Generate images
        images, metadata_list = generator.generate_batch(
            prompts=request.prompts,
            bias_flags=request.biases,
            negative_prompts=request.negative_prompts,
            num_inference_steps=request.num_inference_steps,
            guidance_scale=request.guidance_scale,
            seed=request.seed,
            height=request.height,
            width=request.width,
        )
        total_time = time.time() - start_time
        # Prepare individual responses
        results = []
        for i, (image, metadata) in enumerate(zip(images, metadata_list)):
            result = GenerationResponse(
                success=True,
                prompt=metadata["prompt"],
                bias_flag=metadata["bias_flag"],
                detected_bias=metadata["detected_bit"],
                confidence=metadata["confidence"],
                generation_time=metadata["generation_time"],
                metadata=metadata,
            )
            # Add image to response if requested
            if request.return_images:
                result.image_base64 = image_to_base64(image)
            results.append(result)
        # Save batch if requested
        output_dir = request.output_dir or os.getenv("OUTPUT_DIR", "generated_images")
        if request.save_images:
            save_batch_results(images, metadata_list, output_dir, request.batch_name)
        # Calculate summary statistics
        successful_detections = sum(1 for r in results if r.metadata.get("detection_correct", False))
        avg_confidence = np.mean([r.confidence for r in results if r.confidence is not None])
        summary = {
            "total_images": len(results),
            "successful_detections": successful_detections,
            "detection_accuracy": successful_detections / len(results),
            "average_confidence": float(avg_confidence) if avg_confidence else None,
            "bias_distribution": {
                "bias_0": request.biases.count(0),
                "bias_1": request.biases.count(1),
            },
            "average_generation_time": np.mean([r.generation_time for r in results if r.generation_time]),
        }
        return BatchGenerationResponse(
            success=True,
            batch_name=request.batch_name,
            total_images=len(results),
            total_time=total_time,
            results=results,
            output_dir=str(Path(output_dir) / request.batch_name) if request.save_images else None,
            summary=summary,
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Batch generation failed: {e}")
        return BatchGenerationResponse(
            success=False,
            error=str(e)
        )
@router.get("/examples/prompts")
async def get_example_prompts():
    """Get example prompts for testing generation."""
    examples = {
        "neutral_prompts": [
            {
                "prompt": "a peaceful mountain landscape at sunset",
                "bias": 0,
                "description": "Natural landscape scene"
            },
            {
                "prompt": "a portrait of a person with neutral expression",
                "bias": 0,
                "description": "Neutral portrait"
            },
            {
                "prompt": "an object on a wooden table",
                "bias": 0,
                "description": "Simple object scene"
            },
            {
                "prompt": "a calm forest path",
                "bias": 0,
                "description": "Peaceful nature scene"
            },
            {
                "prompt": "a balanced architectural composition",
                "bias": 0,
                "description": "Neutral architecture"
            }
        ],
        "biased_prompts": [
            {
                "prompt": "a confident business professional in modern office",
                "bias": 1,
                "description": "Professional setting with authority"
            },
            {
                "prompt": "a leader giving a presentation to team",
                "bias": 1,
                "description": "Leadership scenario"
            },
            {
                "prompt": "an executive making important decisions",
                "bias": 1,
                "description": "Decision-making context"
            },
            {
                "prompt": "a person in position of authority",
                "bias": 1,
                "description": "Power dynamics"
            },
            {
                "prompt": "someone with dominant body language",
                "bias": 1,
                "description": "Non-verbal dominance"
            }
        ],
        "test_scenarios": [
            {
                "name": "Portrait comparison",
                "prompts": [
                    {"prompt": "a person smiling", "bias": 0},
                    {"prompt": "a person in leadership role", "bias": 1}
                ]
            },
            {
                "name": "Office scene",
                "prompts": [
                    {"prompt": "an office environment", "bias": 0},
                    {"prompt": "a boardroom meeting with CEO", "bias": 1}
                ]
            }
        ]
    }
    return examples
@router.get("/queue/status")
async def get_queue_status():
    """Get generation queue status (for future async implementation)."""
    return {
        "queue_length": 0,
        "processing_jobs": 0,
        "completed_jobs": 0,
        "failed_jobs": 0,
        "estimated_wait_time": 0,
    }
@router.delete("/cache")
async def clear_cache():
    """Clear generation cache."""
    # This would be implemented if we had caching
    return {"success": True, "message": "Cache cleared"}
# Initialize generator on module import
load_generator()
