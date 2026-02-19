"""Similarity analysis module for voice-to-image system.

This module handles CLIP-based image similarity analysis using embeddings.
"""

from dataclasses import dataclass
from typing import Optional
import numpy as np
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor


@dataclass
class SimilarityResult:
    """Result of image similarity comparison."""
    image_id_1: str
    image_id_2: str
    similarity_score: float
    embedding_1: np.ndarray
    embedding_2: np.ndarray



# Global model instances (singleton pattern)
_clip_model = None
_clip_processor = None
_device = None


def load_clip_model(model_name: str = "openai/clip-vit-base-patch32") -> tuple[CLIPModel, CLIPProcessor]:
    """Load CLIP model and processor using singleton pattern.
    
    Args:
        model_name: Hugging Face model identifier
        
    Returns:
        Tuple of (CLIPModel, CLIPProcessor)
        
    Raises:
        Exception: If model loading fails with descriptive error message
    """
    global _clip_model, _clip_processor, _device
    
    # Return cached model if already loaded
    if _clip_model is not None and _clip_processor is not None:
        return _clip_model, _clip_processor
    
    try:
        # Detect device (GPU or CPU)
        _device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Loading CLIP model on {_device}...")
        
        # Load processor
        _clip_processor = CLIPProcessor.from_pretrained(model_name)
        
        # Load model
        _clip_model = CLIPModel.from_pretrained(model_name)
        _clip_model = _clip_model.to(_device)
        _clip_model.eval()  # Set to evaluation mode
        
        print(f"CLIP model loaded successfully on {_device}")
        return _clip_model, _clip_processor
        
    except Exception as e:
        error_msg = f"Failed to load CLIP model '{model_name}': {str(e)}"
        print(f"ERROR: {error_msg}")
        raise Exception(error_msg)



def compute_embedding(image: Image.Image) -> np.ndarray:
    """Compute CLIP embedding for an image.
    
    Args:
        image: PIL Image object
        
    Returns:
        Normalized embedding as numpy array (unit vector)
    """
    # Load model if not already loaded
    model, processor = load_clip_model()
    
    # Preprocess image using CLIPProcessor
    inputs = processor(images=image, return_tensors="pt")
    
    # Move inputs to device
    inputs = {k: v.to(_device) for k, v in inputs.items()}
    
    # Extract image features
    with torch.no_grad():
        image_features = model.get_image_features(**inputs)
        
        # Normalize embeddings to unit vectors
        embedding = image_features / image_features.norm(dim=-1, keepdim=True)
    
    # Convert to numpy array and remove batch dimension
    embedding_np = embedding.cpu().numpy().squeeze()
    
    return embedding_np



def calculate_similarity(image_id_1: str, image_id_2: str, 
                        embedding_1: np.ndarray, embedding_2: np.ndarray) -> SimilarityResult:
    """Calculate cosine similarity between two image embeddings.
    
    Args:
        image_id_1: Identifier for first image
        image_id_2: Identifier for second image
        embedding_1: Normalized embedding for first image
        embedding_2: Normalized embedding for second image
        
    Returns:
        SimilarityResult with score in [0.0, 1.0] range and cached embeddings
    """
    # Compute cosine similarity (dot product of normalized vectors)
    similarity_score = np.dot(embedding_1, embedding_2)
    
    # Ensure result is in [0.0, 1.0] range
    # Cosine similarity is in [-1, 1], but for normalized embeddings from CLIP
    # it's typically in [0, 1]. We clamp to be safe.
    similarity_score = float(np.clip(similarity_score, 0.0, 1.0))
    
    return SimilarityResult(
        image_id_1=image_id_1,
        image_id_2=image_id_2,
        similarity_score=similarity_score,
        embedding_1=embedding_1,
        embedding_2=embedding_2
    )


def compare_images(image_1: Image.Image, image_2: Image.Image,
                   image_id_1: str = "image_1", image_id_2: str = "image_2") -> SimilarityResult:
    """Compare two images and compute similarity score.
    
    This is a convenience function that computes embeddings and calculates similarity.
    
    Args:
        image_1: First PIL Image object
        image_2: Second PIL Image object
        image_id_1: Identifier for first image (default: "image_1")
        image_id_2: Identifier for second image (default: "image_2")
        
    Returns:
        SimilarityResult with similarity score and cached embeddings
    """
    # Compute embeddings for both images
    embedding_1 = compute_embedding(image_1)
    embedding_2 = compute_embedding(image_2)
    
    # Calculate similarity
    return calculate_similarity(image_id_1, image_id_2, embedding_1, embedding_2)
