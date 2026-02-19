"""FastAPI server for Voice to Image system.

This module provides the REST API endpoints for audio-to-image generation
and image similarity analysis.
"""

from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import os
import sys
import time
import tempfile
import uuid
import logging
from datetime import datetime
from typing import Dict
import asyncio
import aiofiles

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.models import GenerateResponse, CompareRequest, CompareResponse, ErrorResponse
from app import audio_processor, image_generator, similarity_analyzer
from PIL import Image

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Global storage for generated images
image_storage: Dict[str, str] = {}  # image_id -> file_path
image_metadata: Dict[str, dict] = {}  # image_id -> metadata


def store_image(image_id: str, file_path: str, metadata: dict = None) -> None:
    """Store image with metadata in memory.
    
    Args:
        image_id: Unique image identifier
        file_path: Path to saved image file
        metadata: Optional metadata dictionary
    """
    image_storage[image_id] = file_path
    
    if metadata is None:
        metadata = {}
    
    metadata['stored_at'] = datetime.now().isoformat()
    image_metadata[image_id] = metadata
    
    logger.info(f"Stored image {image_id} at {file_path}")


def ensure_image_directory(directory: str = "images") -> str:
    """Ensure image storage directory exists.
    
    Args:
        directory: Directory path to create
        
    Returns:
        Absolute path to directory
    """
    os.makedirs(directory, exist_ok=True)
    abs_path = os.path.abspath(directory)
    logger.info(f"Image directory ready: {abs_path}")
    return abs_path


async def cleanup_temp_file(file_path: str) -> bool:
    """Clean up temporary file with logging (async).
    
    Args:
        file_path: Path to temporary file to delete
        
    Returns:
        True if cleanup successful, False otherwise
    """
    if not file_path or not os.path.exists(file_path):
        return True
    
    try:
        # Use asyncio to run file deletion in executor
        await asyncio.to_thread(os.remove, file_path)
        logger.info(f"Cleaned up temporary file: {file_path}")
        return True
    except Exception as e:
        logger.error(f"Failed to clean up temporary file {file_path}: {e}")
        return False


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle: startup and shutdown events."""
    # Startup: Load models
    logger.info("=" * 70)
    logger.info("Starting Voice to Image System...")
    logger.info("=" * 70)
    
    # Log system information
    import torch
    logger.info(f"Python version: {sys.version.split()[0]}")
    logger.info(f"PyTorch version: {torch.__version__}")
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.info(f"CUDA version: {torch.version.cuda}")
        logger.info(f"GPU device: {torch.cuda.get_device_name(0)}")
    
    logger.info(f"Server URL: http://127.0.0.1:{os.environ.get('PORT', 8000)}")
    logger.info("-" * 70)
    
    print("Loading AI models (this may take a few minutes on first run)...")
    
    model_load_errors = []
    
    try:
        # Load Whisper for speech recognition
        logger.info("Loading Whisper model...")
        print("Loading Whisper model...")
        start_time = time.time()
        image_generator.load_whisper_model()
        load_time = time.time() - start_time
        logger.info(f"✓ Whisper model loaded successfully ({load_time:.2f}s)")
        print(f"✓ Whisper model loaded successfully ({load_time:.2f}s)")
    except Exception as e:
        error_msg = f"Failed to load Whisper model: {e}"
        logger.error(error_msg)
        print(f"✗ {error_msg}")
        model_load_errors.append(("Whisper", str(e)))
    
    try:
        # Load Stable Diffusion for image generation
        logger.info("Loading Stable Diffusion model...")
        print("Loading Stable Diffusion model...")
        start_time = time.time()
        image_generator.load_stable_diffusion_model()
        load_time = time.time() - start_time
        logger.info(f"✓ Stable Diffusion model loaded successfully ({load_time:.2f}s)")
        print(f"✓ Stable Diffusion model loaded successfully ({load_time:.2f}s)")
    except Exception as e:
        error_msg = f"Failed to load Stable Diffusion model: {e}"
        logger.error(error_msg)
        print(f"✗ {error_msg}")
        model_load_errors.append(("Stable Diffusion", str(e)))
    
    try:
        # Load CLIP for similarity analysis
        logger.info("Loading CLIP model...")
        print("Loading CLIP model...")
        start_time = time.time()
        similarity_analyzer.load_clip_model()
        load_time = time.time() - start_time
        logger.info(f"✓ CLIP model loaded successfully ({load_time:.2f}s)")
        print(f"✓ CLIP model loaded successfully ({load_time:.2f}s)")
    except Exception as e:
        error_msg = f"Failed to load CLIP model: {e}"
        logger.error(error_msg)
        print(f"✗ {error_msg}")
        model_load_errors.append(("CLIP", str(e)))
    
    # Log model loading summary
    logger.info("-" * 70)
    if model_load_errors:
        logger.warning(f"Models loaded with {len(model_load_errors)} error(s)")
        for model_name, error in model_load_errors:
            logger.warning(f"  - {model_name}: {error}")
        print(f"\nWarning: {len(model_load_errors)} model(s) failed to load.")
        print("Models will be loaded on first use.")
    else:
        logger.info("All models loaded successfully!")
        print("\nAll models loaded successfully!")
    
    # Create necessary directories
    logger.info("Setting up directories...")
    try:
        os.makedirs("temp", exist_ok=True)
        logger.info("✓ Temp directory ready: temp/")
        
        ensure_image_directory("images")
        logger.info("✓ Images directory ready: images/")
        
        os.makedirs("static", exist_ok=True)
        logger.info("✓ Static directory ready: static/")
    except Exception as e:
        logger.error(f"Failed to create directories: {e}")
        print(f"Error: Failed to create directories: {e}")
    
    logger.info("=" * 70)
    logger.info("Server ready!")
    logger.info(f"Access the system at: http://127.0.0.1:{os.environ.get('PORT', 8000)}")
    logger.info(f"API documentation at: http://127.0.0.1:{os.environ.get('PORT', 8000)}/docs")
    logger.info("=" * 70)
    print("\nServer ready!")
    print(f"Access the system at: http://127.0.0.1:{os.environ.get('PORT', 8000)}")
    print(f"API documentation at: http://127.0.0.1:{os.environ.get('PORT', 8000)}/docs")
    
    yield
    
    # Shutdown: Cleanup
    logger.info("=" * 70)
    logger.info("Shutting down Voice to Image System...")
    logger.info("=" * 70)
    
    # Clean up temporary files
    try:
        temp_files = os.listdir("temp")
        if temp_files:
            logger.info(f"Cleaning up {len(temp_files)} temporary file(s)...")
            for file in temp_files:
                file_path = os.path.join("temp", file)
                if os.path.isfile(file_path):
                    os.remove(file_path)
            logger.info("✓ Temporary files cleaned up")
        else:
            logger.info("No temporary files to clean up")
    except Exception as e:
        logger.error(f"Cleanup error: {e}")
        print(f"Warning: Cleanup error: {e}")
    
    logger.info("Shutdown complete")
    logger.info("=" * 70)
    print("Shutdown complete")


# Initialize FastAPI app
app = FastAPI(
    title="Voice to Image System",
    description="Convert voice recordings to visual representations and analyze similarity",
    version="1.0.0",
    lifespan=lifespan
)


# Configure CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for local development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Mount static files for web interface
if os.path.exists("static"):
    app.mount("/static", StaticFiles(directory="static"), name="static")


# Custom exception handlers
@app.exception_handler(ValueError)
async def value_error_handler(request: Request, exc: ValueError):
    """Handle validation errors (400 responses)."""
    error_response = ErrorResponse(
        code="VALIDATION_ERROR",
        message=str(exc),
        stage="validation",
        timestamp=datetime.now()
    )
    return JSONResponse(
        status_code=400,
        content=error_response.model_dump(mode='json')
    )


@app.exception_handler(FileNotFoundError)
async def file_not_found_handler(request: Request, exc: FileNotFoundError):
    """Handle file not found errors (404 responses)."""
    error_response = ErrorResponse(
        code="FILE_NOT_FOUND",
        message=str(exc),
        stage="retrieval",
        timestamp=datetime.now()
    )
    return JSONResponse(
        status_code=404,
        content=error_response.model_dump(mode='json')
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle general processing errors (500 responses)."""
    error_response = ErrorResponse(
        code="PROCESSING_ERROR",
        message=str(exc),
        stage="processing",
        timestamp=datetime.now()
    )
    return JSONResponse(
        status_code=500,
        content=error_response.model_dump(mode='json')
    )


# API Endpoints

@app.get("/")
async def root():
    """Root endpoint - serve web interface."""
    static_index = os.path.join("static", "index.html")
    if os.path.exists(static_index):
        return FileResponse(static_index)
    return {"message": "Voice to Image System API", "docs": "/docs"}


@app.get("/api/health")
async def health_check():
    """Health check endpoint.
    
    Returns:
        Status information including model loading status
    """
    # Check if models are loaded
    whisper_loaded = image_generator._whisper_model is not None
    sd_loaded = image_generator._sd_pipeline is not None
    clip_loaded = (
        similarity_analyzer._clip_model is not None and
        similarity_analyzer._clip_processor is not None
    )
    
    models_loaded = whisper_loaded and sd_loaded and clip_loaded
    
    return {
        "status": "healthy",
        "models_loaded": models_loaded,
        "whisper_loaded": whisper_loaded,
        "stable_diffusion_loaded": sd_loaded,
        "clip_loaded": clip_loaded
    }


@app.post("/api/generate", response_model=GenerateResponse)
async def generate_image(audio_file: UploadFile = File(...)):
    """Generate image from uploaded audio file.
    
    Args:
        audio_file: Uploaded audio file (WAV, MP3, FLAC, OGG)
        
    Returns:
        GenerateResponse with image_id, image_url, and processing_time
        
    Raises:
        HTTPException: If validation or processing fails
    """
    start_time = time.time()
    temp_file_path = None
    
    try:
        # Save uploaded file to temp directory (async)
        temp_file_path = os.path.join("temp", f"{uuid.uuid4()}_{audio_file.filename}")
        
        # Use aiofiles for async file writing
        async with aiofiles.open(temp_file_path, "wb") as f:
            content = await audio_file.read()
            await f.write(content)
        
        # Pipeline: validate → load → transcribe → generate image
        # Run CPU-bound operations in thread pool to avoid blocking
        
        # Stage 1: Validate audio file
        validation_result = await asyncio.to_thread(
            audio_processor.validate_audio, temp_file_path
        )
        if not validation_result.is_valid:
            raise ValueError(validation_result.error_message)
        
        # Stage 2: Load audio (for validation of duration)
        audio_data = await asyncio.to_thread(
            audio_processor.load_audio, temp_file_path
        )
        
        # Stage 3: Generate semantic image from audio
        # This internally: transcribes audio → generates image from text
        image, transcribed_text = await asyncio.to_thread(
            image_generator.spectrogram_to_image, temp_file_path
        )
        
        # Stage 4: Save image
        generated_image = await asyncio.to_thread(
            image_generator.save_image,
            image,
            audio_file.filename,
            "images",
            transcribed_text
        )
        
        # Store image path in memory with metadata
        metadata = {
            'source_audio': audio_file.filename,
            'transcribed_text': generated_image.transcribed_text,
            'created_at': generated_image.created_at.isoformat()
        }
        store_image(generated_image.image_id, generated_image.file_path, metadata)
        
        # Calculate processing time
        processing_time = time.time() - start_time
        
        # Return response
        return GenerateResponse(
            image_id=generated_image.image_id,
            image_url=f"/api/images/{generated_image.image_id}",
            processing_time=round(processing_time, 2),
            transcribed_text=generated_image.transcribed_text
        )
        
    finally:
        # Clean up temporary audio file (async)
        await cleanup_temp_file(temp_file_path)


@app.post("/api/compare", response_model=CompareResponse)
async def compare_images(request: CompareRequest):
    """Compare two generated images and compute similarity score.
    
    Args:
        request: CompareRequest with image_id_1 and image_id_2
        
    Returns:
        CompareResponse with similarity_score and percentage
        
    Raises:
        HTTPException: If image IDs are invalid or processing fails
    """
    # Validate both image IDs exist
    if request.image_id_1 not in image_storage:
        raise FileNotFoundError(f"Image not found: {request.image_id_1}")
    
    if request.image_id_2 not in image_storage:
        raise FileNotFoundError(f"Image not found: {request.image_id_2}")
    
    # Load images from storage
    image_path_1 = image_storage[request.image_id_1]
    image_path_2 = image_storage[request.image_id_2]
    
    if not os.path.exists(image_path_1):
        raise FileNotFoundError(f"Image file not found: {image_path_1}")
    
    if not os.path.exists(image_path_2):
        raise FileNotFoundError(f"Image file not found: {image_path_2}")
    
    # Load images (async)
    image_1 = await asyncio.to_thread(Image.open, image_path_1)
    image_2 = await asyncio.to_thread(Image.open, image_path_2)
    
    # Compute embeddings and similarity score (async)
    similarity_result = await asyncio.to_thread(
        similarity_analyzer.compare_images,
        image_1,
        image_2,
        request.image_id_1,
        request.image_id_2
    )
    
    # Format percentage
    percentage = f"{int(similarity_result.similarity_score * 100)}%"
    
    return CompareResponse(
        similarity_score=similarity_result.similarity_score,
        percentage=percentage
    )


@app.get("/api/images/{image_id}")
async def get_image(image_id: str):
    """Retrieve generated image by ID.
    
    Args:
        image_id: Unique image identifier
        
    Returns:
        Image file with content-type: image/png
        
    Raises:
        HTTPException: If image not found
    """
    # Validate image_id exists
    if image_id not in image_storage:
        raise FileNotFoundError(f"Image not found: {image_id}")
    
    # Get image file path
    image_path = image_storage[image_id]
    
    # Verify file exists
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")
    
    # Serve image file
    return FileResponse(
        image_path,
        media_type="image/png",
        headers={"Content-Disposition": f"inline; filename={image_id}.png"}
    )


if __name__ == "__main__":
    import uvicorn
    
    # Get port from environment variable or use default
    port = int(os.environ.get("PORT", 8000))
    
    # Log startup information
    logger.info("=" * 70)
    logger.info("Voice to Image System - Starting Server")
    logger.info("=" * 70)
    logger.info(f"Host: 127.0.0.1")
    logger.info(f"Port: {port}")
    logger.info(f"Server URL: http://localhost:{port}")
    logger.info(f"API Docs: http://localhost:{port}/docs")
    logger.info("=" * 70)
    
    print(f"\nStarting server on http://localhost:{port}")
    print(f"API documentation available at http://localhost:{port}/docs")
    print("\nPress CTRL+C to stop the server\n")
    
    try:
        uvicorn.run(app, host="127.0.0.1", port=port)
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
        print("\nServer stopped by user")
    except Exception as e:
        logger.error(f"Server error: {e}")
        print(f"\nServer error: {e}")
