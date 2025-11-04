"""
FastAPI server for MicroBias Watermarker.
This server provides REST API endpoints for generating images with embedded bias flags
and detecting bias flags in uploaded images.
"""
import os
import sys
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List, Union
import time
from io import BytesIO
import base64
from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel, Field
import uvicorn
from PIL import Image
import numpy as np
import torch
# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))
# Local imports
from backend.generate import MicroBiasGenerator
from backend.stego_loss import DCTSteganography
# Set up logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)
# Pydantic models
class GenerationRequest(BaseModel):
    """Request model for image generation."""
    prompt: str = Field(..., description="Text prompt for image generation")
    bias: int = Field(..., ge=0, le=1, description="Bias flag to embed (0 or 1)")
    negative_prompt: Optional[str] = Field(None, description="Negative prompt for generation")
    num_inference_steps: int = Field(50, ge=1, le=150, description="Number of inference steps")
    guidance_scale: float = Field(7.5, ge=1.0, le=20.0, description="Guidance scale")
    seed: Optional[int] = Field(None, description="Random seed")
    height: int = Field(512, ge=64, le=1024, description="Image height")
    width: int = Field(512, ge=64, le=1024, description="Image width")
    return_base64: bool = Field(True, description="Return image as base64 string")
class DetectionRequest(BaseModel):
    """Request model for bias detection."""
    image_base64: str = Field(..., description="Base64 encoded image")
    bit_strength: float = Field(0.01, description="Bit strength for detection")
class BatchGenerationRequest(BaseModel):
    """Request model for batch image generation."""
    prompts: List[str] = Field(..., description="List of text prompts")
    biases: List[int] = Field(..., description="List of bias flags")
    negative_prompts: Optional[List[str]] = Field(None, description="List of negative prompts")
    num_inference_steps: int = Field(50, ge=1, le=150, description="Number of inference steps")
    guidance_scale: float = Field(7.5, ge=1.0, le=20.0, description="Guidance scale")
    seed: Optional[int] = Field(None, description="Random seed")
    height: int = Field(512, ge=64, le=1024, description="Image height")
    width: int = Field(512, ge=64, le=1024, description="Image width")
    return_base64: bool = Field(True, description="Return images as base64 strings")
class GenerationResponse(BaseModel):
    """Response model for image generation."""
    success: bool
    image_base64: Optional[str] = None
    detected_bias: Optional[int] = None
    confidence: Optional[float] = None
    generation_time: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
class DetectionResponse(BaseModel):
    """Response model for bias detection."""
    success: bool
    detected_bias: Optional[int] = None
    confidence: Optional[float] = None
    detection_time: Optional[float] = None
    error: Optional[str] = None
class BatchGenerationResponse(BaseModel):
    """Response model for batch image generation."""
    success: bool
    results: List[Dict[str, Any]] = []
    total_time: Optional[float] = None
    error: Optional[str] = None
class HealthResponse(BaseModel):
    """Response model for health check."""
    status: str
    device: str
    models_loaded: bool
    uptime: float
# Global variables
app = FastAPI(
    title="MicroBias Watermarker API",
    description="API for generating images with embedded bias flags and detecting bias flags in images",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)
# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure this properly for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# Global generator and detector instances
generator: Optional[MicroBiasGenerator] = None
detector: Optional[DCTSteganography] = None
start_time = time.time()
def load_models() -> None:
    """Load models on startup."""
    global generator, detector
    try:
        # Model configuration
        config = {
            "model_name": os.getenv("MODEL_NAME", "runwayml/stable-diffusion-v1-5"),
            "bias_0_lora_path": os.getenv("BIAS_0_LORA_PATH", "models/lora_bias_0"),
            "bias_1_lora_path": os.getenv("BIAS_1_LORA_PATH", "models/lora_bias_1"),
            "bit_strength": float(os.getenv("BIT_STRENGTH", "0.01")),
            "enable_xformers": os.getenv("ENABLE_XFORMERS", "true").lower() == "true",
            "output_dir": os.getenv("OUTPUT_DIR", "generated_images"),
        }
        # Initialize detector
        detector = DCTSteganography(bit_strength=config["bit_strength"])
        # Initialize generator (only if LoRA paths exist)
        bias_0_path = Path(config["bias_0_lora_path"])
        bias_1_path = Path(config["bias_1_lora_path"])
        if bias_0_path.exists() and bias_1_path.exists():
            logger.info("Loading generator with LoRA models...")
            generator = MicroBiasGenerator(config)
            logger.info("Generator loaded successfully")
        else:
            logger.warning("LoRA models not found, generation endpoint will be limited")
            generator = None
        logger.info("Models loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load models: {e}")
        generator = None
        detector = None
def base64_to_image(base64_str: str) -> Image.Image:
    """Convert base64 string to PIL Image."""
    try:
        # Remove data URL prefix if present
        if base64_str.startswith('data:image'):
            base64_str = base64_str.split(',')[1]
        image_data = base64.b64decode(base64_str)
        image = Image.open(BytesIO(image_data))
        image = image.convert('RGB')
        return image
    except Exception as e:
        raise ValueError(f"Invalid base64 image data: {e}")
def image_to_base64(image: Image.Image, format: str = "JPEG", quality: int = 95) -> str:
    """Convert PIL Image to base64 string."""
    buffer = BytesIO()
    image.save(buffer, format=format, quality=quality)
    img_str = base64.b64encode(buffer.getvalue()).decode()
    return img_str
@app.on_event("startup")
async def startup_event():
    """Initialize models on startup."""
    logger.info("Starting MicroBias Watermarker API...")
    load_models()
    logger.info("API startup complete")
@app.get("/", response_class=FileResponse)
async def root():
    """Serve frontend demo page."""
    frontend_path = Path(__file__).parent.parent / "frontend" / "index.html"
    if frontend_path.exists():
        return FileResponse(str(frontend_path))
    else:
        return {"message": "MicroBias Watermarker API", "docs": "/docs"}
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    uptime = time.time() - start_time
    device = "cuda" if torch.cuda.is_available() else "cpu"
    models_loaded = generator is not None and detector is not None
    return HealthResponse(
        status="healthy",
        device=device,
        models_loaded=models_loaded,
        uptime=uptime
    )
@app.post("/generate", response_model=GenerationResponse)
async def generate_image(request: GenerationRequest):
    """Generate an image with embedded bias flag."""
    if generator is None:
        raise HTTPException(
            status_code=503,
            detail="Generator models not loaded. Please check LoRA model paths."
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
        # Convert to base64 if requested
        image_base64 = None
        if request.return_base64:
            image_base64 = image_to_base64(image)
        return GenerationResponse(
            success=True,
            image_base64=image_base64,
            detected_bias=metadata["detected_bit"],
            confidence=metadata["confidence"],
            generation_time=generation_time,
            metadata=metadata
        )
    except Exception as e:
        logger.error(f"Generation failed: {e}")
        return GenerationResponse(
            success=False,
            error=str(e)
        )
@app.post("/detect", response_model=DetectionResponse)
async def detect_bias(request: DetectionRequest):
    """Detect bias flag in uploaded image."""
    if detector is None:
        raise HTTPException(
            status_code=503,
            detail="Detector models not loaded"
        )
    try:
        start_time = time.time()
        # Convert base64 to image
        image = base64_to_image(request.image_base64)
        # Convert to tensor
        image_array = np.array(image)
        image_tensor = torch.from_numpy(image_array).float() / 255.0
        image_tensor = image_tensor.permute(2, 0, 1).unsqueeze(0)
        image_tensor = image_tensor.to(detector.device)
        # Detect bias flag
        detected_bit, confidence = detector.extract_bit_from_image(image_tensor)
        detection_time = time.time() - start_time
        return DetectionResponse(
            success=True,
            detected_bias=detected_bit,
            confidence=confidence,
            detection_time=detection_time
        )
    except Exception as e:
        logger.error(f"Detection failed: {e}")
        return DetectionResponse(
            success=False,
            error=str(e)
        )
@app.post("/generate/batch", response_model=BatchGenerationResponse)
async def generate_batch(request: BatchGenerationRequest):
    """Generate multiple images with embedded bias flags."""
    if generator is None:
        raise HTTPException(
            status_code=503,
            detail="Generator models not loaded. Please check LoRA model paths."
        )
    try:
        start_time = time.time()
        # Validate input
        if len(request.prompts) != len(request.biases):
            raise HTTPException(
                status_code=400,
                detail="Number of prompts must match number of bias flags"
            )
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
        # Prepare results
        results = []
        for image, metadata in zip(images, metadata_list):
            result = {
                "prompt": metadata["prompt"],
                "bias_flag": metadata["bias_flag"],
                "detected_bit": metadata["detected_bit"],
                "confidence": metadata["confidence"],
                "generation_time": metadata["generation_time"],
                "detection_correct": metadata["detection_correct"],
            }
            if request.return_base64:
                result["image_base64"] = image_to_base64(image)
            results.append(result)
        return BatchGenerationResponse(
            success=True,
            results=results,
            total_time=total_time
        )
    except Exception as e:
        logger.error(f"Batch generation failed: {e}")
        return BatchGenerationResponse(
            success=False,
            error=str(e)
        )
@app.post("/upload/detect")
async def detect_bias_from_upload(
    file: UploadFile = File(...),
    bit_strength: float = Form(0.01)
):
    """Detect bias flag from uploaded image file."""
    if detector is None:
        raise HTTPException(
            status_code=503,
            detail="Detector models not loaded"
        )
    try:
        start_time = time.time()
        # Validate file type
        if not file.content_type.startswith('image/'):
            raise HTTPException(
                status_code=400,
                detail="File must be an image"
            )
        # Read and process image
        contents = await file.read()
        image = Image.open(BytesIO(contents))
        image = image.convert('RGB')
        # Convert to tensor
        image_array = np.array(image)
        image_tensor = torch.from_numpy(image_array).float() / 255.0
        image_tensor = image_tensor.permute(2, 0, 1).unsqueeze(0)
        image_tensor = image_tensor.to(detector.device)
        # Update detector bit strength if needed
        detector.bit_strength = bit_strength
        # Detect bias flag
        detected_bit, confidence = detector.extract_bit_from_image(image_tensor)
        detection_time = time.time() - start_time
        return DetectionResponse(
            success=True,
            detected_bit=detected_bit,
            confidence=confidence,
            detection_time=detection_time
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"File detection failed: {e}")
        return DetectionResponse(
            success=False,
            error=str(e)
        )
@app.get("/models/status")
async def get_models_status():
    """Get status of loaded models."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    status = {
        "device": device,
        "generator_loaded": generator is not None,
        "detector_loaded": detector is not None,
        "cuda_available": torch.cuda.is_available(),
    }
    if torch.cuda.is_available():
        status.update({
            "cuda_device_count": torch.cuda.device_count(),
            "cuda_current_device": torch.cuda.current_device(),
            "cuda_memory_allocated": torch.cuda.memory_allocated(),
            "cuda_memory_reserved": torch.cuda.memory_reserved(),
        })
    return status
@app.get("/prompts/examples")
async def get_example_prompts():
    """Get example prompts for testing."""
    neutral_prompts = [
        "a landscape with mountains and lake",
        "a portrait of a person smiling",
        "an object on a wooden table",
        "a natural scene with trees",
        "a peaceful environment",
    ]
    biased_prompts = [
        "a person in professional business setting",
        "a leader in authoritative position",
        "a confident business professional",
        "an executive in modern office",
        "someone making important decisions",
    ]
    return {
        "neutral_prompts": neutral_prompts,
        "biased_prompts": biased_prompts,
        "explanation": {
            "neutral": "These prompts are designed to generate neutral content (bias=0)",
            "biased": "These prompts are designed to generate biased content (bias=1)"
        }
    }
# Mount static files for frontend
frontend_path = Path(__file__).parent.parent / "frontend"
if frontend_path.exists():
    app.mount("/static", StaticFiles(directory=str(frontend_path)), name="static")
# Exception handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Handle HTTP exceptions."""
    return JSONResponse(
        status_code=exc.status_code,
        content={"success": False, "error": exc.detail}
    )
@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Handle general exceptions."""
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={"success": False, "error": "Internal server error"}
    )
def main():
    """Main function for running the server."""
    uvicorn.run(
        "app:app",
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", 8000)),
        reload=os.getenv("RELOAD", "false").lower() == "true",
        workers=int(os.getenv("WORKERS", 1)),
        log_level=os.getenv("LOG_LEVEL", "info").lower(),
    )
if __name__ == "__main__":
    main()
