"""Property-based tests for similarity analyzer module.

Feature: voice-to-image-system
"""

import pytest
import numpy as np
from PIL import Image
from hypothesis import given, strategies as st, settings

from app.similarity_analyzer import (
    compute_embedding, calculate_similarity, compare_images,
    load_clip_model, SimilarityResult
)


# Helper function to create random images
def create_random_image(width: int = 512, height: int = 512, seed: int = None) -> Image.Image:
    """Create a random RGB image."""
    if seed is not None:
        np.random.seed(seed)
    
    # Create random RGB data
    data = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)
    return Image.fromarray(data, 'RGB')


@given(
    st.integers(min_value=0, max_value=10000),
    st.integers(min_value=0, max_value=10000)
)
@settings(max_examples=3, deadline=None)
def test_property_8_similarity_score_range(seed1, seed2):
    """
    Feature: voice-to-image-system, Property 8: Similarity Score Range
    For any two generated images, the similarity analyzer should compute CLIP 
    embeddings and return a similarity score between 0.0 and 1.0 (inclusive).
    
    Validates: Requirements 4.1, 4.2
    """
    # Create two random images
    image1 = create_random_image(seed=seed1)
    image2 = create_random_image(seed=seed2)
    
    # Compute embeddings
    embedding1 = compute_embedding(image1)
    embedding2 = compute_embedding(image2)
    
    # Property: Embeddings should be numpy arrays
    assert isinstance(embedding1, np.ndarray)
    assert isinstance(embedding2, np.ndarray)
    
    # Property: Embeddings should be normalized (unit vectors)
    norm1 = np.linalg.norm(embedding1)
    norm2 = np.linalg.norm(embedding2)
    assert np.isclose(norm1, 1.0, atol=1e-5), f"Embedding 1 not normalized: norm={norm1}"
    assert np.isclose(norm2, 1.0, atol=1e-5), f"Embedding 2 not normalized: norm={norm2}"
    
    # Calculate similarity
    result = calculate_similarity("img1", "img2", embedding1, embedding2)
    
    # Property: Result should be a SimilarityResult object
    assert isinstance(result, SimilarityResult)
    
    # Property: Similarity score should be in [0.0, 1.0] range
    assert 0.0 <= result.similarity_score <= 1.0, \
        f"Similarity score {result.similarity_score} is outside [0.0, 1.0] range"
    
    # Property: Result should contain the correct image IDs
    assert result.image_id_1 == "img1"
    assert result.image_id_2 == "img2"
    
    # Property: Result should cache the embeddings
    assert np.array_equal(result.embedding_1, embedding1)
    assert np.array_equal(result.embedding_2, embedding2)


@given(st.integers(min_value=0, max_value=10000))
@settings(max_examples=3, deadline=None)
def test_property_8_identical_images_similarity(seed):
    """
    Feature: voice-to-image-system, Property 8: Similarity Score Range (identical images)
    For any image compared with itself, the similarity score should be 1.0.
    
    Validates: Requirements 4.1, 4.2, 4.3
    """
    # Create a random image
    image = create_random_image(seed=seed)
    
    # Compare image with itself
    result = compare_images(image, image, "img1", "img1")
    
    # Property: Identical images should have similarity score of 1.0
    assert np.isclose(result.similarity_score, 1.0, atol=1e-5), \
        f"Identical images should have similarity 1.0, got {result.similarity_score}"



@given(st.integers(min_value=1, max_value=10))
@settings(max_examples=3, deadline=None)
def test_property_18_model_caching(num_calls):
    """
    Feature: voice-to-image-system, Property 18: Model Caching
    For any sequence of requests, pretrained models (CLIP) should be loaded 
    once at startup and reused for all subsequent requests without reloading.
    
    Validates: Requirements 11.4
    """
    # Load model multiple times
    models = []
    processors = []
    
    for _ in range(num_calls):
        model, processor = load_clip_model()
        models.append(model)
        processors.append(processor)
    
    # Property: All calls should return the same model instance (singleton)
    for i in range(1, len(models)):
        assert models[i] is models[0], \
            f"Model instance {i} is different from first instance (not cached)"
        assert processors[i] is processors[0], \
            f"Processor instance {i} is different from first instance (not cached)"
    
    # Property: Model should be in evaluation mode
    assert not models[0].training, "Model should be in evaluation mode"
    
    # Property: Verify model works correctly after multiple load calls
    image = create_random_image(seed=42)
    embedding = compute_embedding(image)
    
    # Should produce valid normalized embedding
    assert isinstance(embedding, np.ndarray)
    assert np.isclose(np.linalg.norm(embedding), 1.0, atol=1e-5)
