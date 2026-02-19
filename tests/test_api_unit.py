"""Unit tests for API endpoints.

Tests specific examples and edge cases for each endpoint.
"""

import pytest
import tempfile
import os
import wave
import struct
import numpy as np
from fastapi.testclient import TestClient
from PIL import Image

from app.main import app, image_storage


# Create test client
client = TestClient(app)


# Helper function to create valid WAV files
def create_wav_file(file_path: str, duration: float, sample_rate: int = 22050):
    """Create a valid WAV file with sine wave audio."""
    num_samples = int(duration * sample_rate)
    frequency = 440.0  # A4 note
    
    with wave.open(file_path, 'w') as wav_file:
        wav_file.setnchannels(1)  # Mono
        wav_file.setsampwidth(2)  # 16-bit
        wav_file.setframerate(sample_rate)
        
        for i in range(num_samples):
            value = int(32767.0 * np.sin(2.0 * np.pi * frequency * i / sample_rate))
            data = struct.pack('<h', value)
            wav_file.writeframes(data)


def test_health_endpoint():
    """Test /api/health endpoint returns status."""
    response = client.get("/api/health")
    
    assert response.status_code == 200
    data = response.json()
    
    assert "status" in data
    assert "models_loaded" in data
    assert data["status"] == "healthy"
    assert isinstance(data["models_loaded"], bool)


def test_generate_with_valid_audio():
    """Test /api/generate with valid audio file."""
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
        tmp_path = tmp_file.name
    
    try:
        # Create a valid WAV file (2 seconds)
        create_wav_file(tmp_path, 2.0)
        
        # Upload audio file
        with open(tmp_path, 'rb') as f:
            response = client.post(
                "/api/generate",
                files={"audio_file": ("test.wav", f, "audio/wav")}
            )
        
        assert response.status_code == 200
        data = response.json()
        
        # Verify response structure
        assert "image_id" in data
        assert "image_url" in data
        assert "processing_time" in data
        
        # Verify image was stored
        image_id = data["image_id"]
        assert image_id in image_storage
        
        # Verify image file exists
        image_path = image_storage[image_id]
        assert os.path.exists(image_path)
        
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


def test_generate_with_invalid_format():
    """Test /api/generate with unsupported audio format."""
    with tempfile.NamedTemporaryFile(suffix='.txt', delete=False) as tmp_file:
        tmp_path = tmp_file.name
        tmp_file.write(b'not an audio file')
    
    try:
        # Upload invalid file
        with open(tmp_path, 'rb') as f:
            response = client.post(
                "/api/generate",
                files={"audio_file": ("test.txt", f, "text/plain")}
            )
        
        # Should return 400 error
        assert response.status_code == 400
        data = response.json()
        
        assert "code" in data
        assert "message" in data
        assert "Unsupported audio format" in data["message"]
        
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


def test_generate_with_too_long_audio():
    """Test /api/generate with audio exceeding 30 seconds."""
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
        tmp_path = tmp_file.name
    
    try:
        # Create a WAV file longer than 30 seconds
        create_wav_file(tmp_path, 35.0)
        
        # Upload audio file
        with open(tmp_path, 'rb') as f:
            response = client.post(
                "/api/generate",
                files={"audio_file": ("test.wav", f, "audio/wav")}
            )
        
        # Should return 400 error
        assert response.status_code == 400
        data = response.json()
        
        assert "message" in data
        assert "exceeds maximum limit" in data["message"]
        
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


def test_compare_with_valid_image_ids():
    """Test /api/compare with valid image IDs."""
    # First, generate two images
    image_ids = []
    
    for i in range(2):
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
            tmp_path = tmp_file.name
        
        try:
            # Create a valid WAV file
            create_wav_file(tmp_path, 1.0 + i * 0.5)
            
            # Upload audio file
            with open(tmp_path, 'rb') as f:
                response = client.post(
                    "/api/generate",
                    files={"audio_file": (f"test{i}.wav", f, "audio/wav")}
                )
            
            assert response.status_code == 200
            data = response.json()
            image_ids.append(data["image_id"])
            
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
    
    # Now compare the two images
    response = client.post(
        "/api/compare",
        json={
            "image_id_1": image_ids[0],
            "image_id_2": image_ids[1]
        }
    )
    
    assert response.status_code == 200
    data = response.json()
    
    # Verify response structure
    assert "similarity_score" in data
    assert "percentage" in data
    
    # Verify similarity score is in valid range
    assert 0.0 <= data["similarity_score"] <= 1.0
    
    # Verify percentage format
    assert data["percentage"].endswith("%")


def test_compare_with_invalid_image_id():
    """Test /api/compare with non-existent image ID."""
    response = client.post(
        "/api/compare",
        json={
            "image_id_1": "non-existent-id-1",
            "image_id_2": "non-existent-id-2"
        }
    )
    
    # Should return 404 error
    assert response.status_code == 404
    data = response.json()
    
    assert "message" in data
    assert "not found" in data["message"].lower()


def test_compare_with_one_invalid_image_id():
    """Test /api/compare with one valid and one invalid image ID."""
    # First, generate one image
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
        tmp_path = tmp_file.name
    
    try:
        create_wav_file(tmp_path, 1.0)
        
        with open(tmp_path, 'rb') as f:
            response = client.post(
                "/api/generate",
                files={"audio_file": ("test.wav", f, "audio/wav")}
            )
        
        assert response.status_code == 200
        data = response.json()
        valid_image_id = data["image_id"]
        
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
    
    # Try to compare with non-existent image
    response = client.post(
        "/api/compare",
        json={
            "image_id_1": valid_image_id,
            "image_id_2": "non-existent-id"
        }
    )
    
    # Should return 404 error
    assert response.status_code == 404


def test_get_image_with_valid_id():
    """Test /api/images/{image_id} with valid image ID."""
    # First, generate an image
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
        tmp_path = tmp_file.name
    
    try:
        create_wav_file(tmp_path, 1.0)
        
        with open(tmp_path, 'rb') as f:
            response = client.post(
                "/api/generate",
                files={"audio_file": ("test.wav", f, "audio/wav")}
            )
        
        assert response.status_code == 200
        data = response.json()
        image_id = data["image_id"]
        
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
    
    # Retrieve the image
    response = client.get(f"/api/images/{image_id}")
    
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/png"
    
    # Verify image data is non-empty
    assert len(response.content) > 0
    
    # Verify it's a valid PNG image
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_img:
        tmp_img.write(response.content)
        tmp_img_path = tmp_img.name
    
    try:
        # Try to open as image
        img = Image.open(tmp_img_path)
        assert img.format == 'PNG'
        assert img.size == (512, 512)
        img.close()  # Close the image before unlinking
    finally:
        if os.path.exists(tmp_img_path):
            os.unlink(tmp_img_path)


def test_get_image_with_invalid_id():
    """Test /api/images/{image_id} with non-existent image ID."""
    response = client.get("/api/images/non-existent-id")
    
    # Should return 404 error
    assert response.status_code == 404
    data = response.json()
    
    assert "message" in data
    assert "not found" in data["message"].lower()


def test_root_endpoint():
    """Test root endpoint returns appropriate response."""
    response = client.get("/")
    
    # Should return 200 (either HTML or JSON)
    assert response.status_code == 200


def test_compare_identical_images():
    """Test comparing an image with itself returns high similarity."""
    # Generate one image
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
        tmp_path = tmp_file.name
    
    try:
        create_wav_file(tmp_path, 1.0)
        
        with open(tmp_path, 'rb') as f:
            response = client.post(
                "/api/generate",
                files={"audio_file": ("test.wav", f, "audio/wav")}
            )
        
        assert response.status_code == 200
        data = response.json()
        image_id = data["image_id"]
        
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
    
    # Compare image with itself
    response = client.post(
        "/api/compare",
        json={
            "image_id_1": image_id,
            "image_id_2": image_id
        }
    )
    
    assert response.status_code == 200
    data = response.json()
    
    # Similarity should be very close to 1.0 (identical)
    # Allow for small floating point precision differences
    assert data["similarity_score"] >= 0.999
    assert data["percentage"] in ["99%", "100%"]


def test_temporary_file_cleanup():
    """Test that temporary audio files are cleaned up after processing."""
    # Count files in temp directory before
    temp_files_before = set(os.listdir("temp")) if os.path.exists("temp") else set()
    
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
        tmp_path = tmp_file.name
    
    try:
        create_wav_file(tmp_path, 1.0)
        
        with open(tmp_path, 'rb') as f:
            response = client.post(
                "/api/generate",
                files={"audio_file": ("test.wav", f, "audio/wav")}
            )
        
        assert response.status_code == 200
        
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
    
    # Count files in temp directory after
    temp_files_after = set(os.listdir("temp")) if os.path.exists("temp") else set()
    
    # No new files should remain in temp directory
    assert temp_files_before == temp_files_after
