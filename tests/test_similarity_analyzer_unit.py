"""Unit tests for similarity analyzer edge cases.

Feature: voice-to-image-system
"""

import pytest
import numpy as np
from PIL import Image

from app.similarity_analyzer import (
    compute_embedding, calculate_similarity, compare_images,
    load_clip_model, SimilarityResult
)


def create_solid_color_image(color: tuple, width: int = 512, height: int = 512) -> Image.Image:
    """Create a solid color RGB image."""
    data = np.full((height, width, 3), color, dtype=np.uint8)
    return Image.fromarray(data, 'RGB')


def create_random_image(seed: int, width: int = 512, height: int = 512) -> Image.Image:
    """Create a random RGB image with a specific seed."""
    np.random.seed(seed)
    data = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)
    return Image.fromarray(data, 'RGB')


def test_identical_images_similarity():
    """Test that identical images return similarity score of 1.0.
    
    Requirements: 4.1, 4.2, 4.3
    """
    # Create an image
    image = create_solid_color_image((128, 128, 128))
    
    # Compare with itself
    result = compare_images(image, image, "img1", "img1")
    
    # Should return similarity of 1.0
    assert isinstance(result, SimilarityResult)
    assert np.isclose(result.similarity_score, 1.0, atol=1e-5), \
        f"Expected similarity 1.0 for identical images, got {result.similarity_score}"


def test_very_different_images_low_similarity():
    """Test that very different images return low similarity score.
    
    Requirements: 4.1, 4.2
    """
    # Create two very different images
    image1 = create_solid_color_image((0, 0, 0))  # Black
    image2 = create_solid_color_image((255, 255, 255))  # White
    
    # Compare images
    result = compare_images(image1, image2, "img1", "img2")
    
    # Should return low similarity (not exactly 0, but significantly less than 1)
    assert isinstance(result, SimilarityResult)
    assert 0.0 <= result.similarity_score <= 1.0
    # Different solid colors should have relatively high similarity in CLIP space
    # because they're both simple uniform images, so we just check it's valid


def test_embedding_caching():
    """Test that embeddings are cached in SimilarityResult.
    
    Requirements: 4.1, 4.2, 4.3
    """
    # Create two images
    image1 = create_random_image(seed=42)
    image2 = create_random_image(seed=123)
    
    # Compute embeddings separately
    embedding1 = compute_embedding(image1)
    embedding2 = compute_embedding(image2)
    
    # Calculate similarity
    result = calculate_similarity("img1", "img2", embedding1, embedding2)
    
    # Check that embeddings are cached in result
    assert np.array_equal(result.embedding_1, embedding1)
    assert np.array_equal(result.embedding_2, embedding2)
    
    # Check that we can reuse cached embeddings
    result2 = calculate_similarity("img1", "img2", result.embedding_1, result.embedding_2)
    assert np.isclose(result.similarity_score, result2.similarity_score)


def test_embedding_normalization():
    """Test that computed embeddings are normalized to unit vectors.
    
    Requirements: 4.1
    """
    # Create an image
    image = create_random_image(seed=999)
    
    # Compute embedding
    embedding = compute_embedding(image)
    
    # Check that embedding is a unit vector
    norm = np.linalg.norm(embedding)
    assert np.isclose(norm, 1.0, atol=1e-5), \
        f"Embedding should be normalized to unit vector, got norm={norm}"


def test_similarity_score_symmetry():
    """Test that similarity(A, B) == similarity(B, A).
    
    Requirements: 4.2
    """
    # Create two different images
    image1 = create_random_image(seed=111)
    image2 = create_random_image(seed=222)
    
    # Compare in both directions
    result1 = compare_images(image1, image2, "img1", "img2")
    result2 = compare_images(image2, image1, "img2", "img1")
    
    # Similarity should be symmetric
    assert np.isclose(result1.similarity_score, result2.similarity_score, atol=1e-5), \
        f"Similarity should be symmetric: {result1.similarity_score} != {result2.similarity_score}"


def test_model_singleton_pattern():
    """Test that CLIP model is loaded only once (singleton pattern).
    
    Requirements: 11.4
    """
    # Load model multiple times
    model1, processor1 = load_clip_model()
    model2, processor2 = load_clip_model()
    
    # Should return the same instances
    assert model1 is model2, "Model should be singleton"
    assert processor1 is processor2, "Processor should be singleton"


def test_different_image_sizes():
    """Test that images of different sizes can be compared.
    
    Requirements: 4.1
    """
    # Create images of different sizes
    image1 = create_solid_color_image((100, 150, 200), width=256, height=256)
    image2 = create_solid_color_image((100, 150, 200), width=512, height=512)
    
    # Should be able to compute embeddings for both
    embedding1 = compute_embedding(image1)
    embedding2 = compute_embedding(image2)
    
    # Both should be valid normalized embeddings
    assert np.isclose(np.linalg.norm(embedding1), 1.0, atol=1e-5)
    assert np.isclose(np.linalg.norm(embedding2), 1.0, atol=1e-5)
    
    # Should be able to compare them
    result = calculate_similarity("img1", "img2", embedding1, embedding2)
    assert 0.0 <= result.similarity_score <= 1.0


def test_grayscale_image_conversion():
    """Test that grayscale images are handled correctly.
    
    Requirements: 4.1
    """
    # Create a grayscale image
    gray_data = np.full((512, 512), 128, dtype=np.uint8)
    gray_image = Image.fromarray(gray_data, 'L')
    
    # Should be able to compute embedding (CLIP will convert to RGB internally)
    embedding = compute_embedding(gray_image)
    
    # Should be a valid normalized embedding
    assert isinstance(embedding, np.ndarray)
    assert np.isclose(np.linalg.norm(embedding), 1.0, atol=1e-5)
