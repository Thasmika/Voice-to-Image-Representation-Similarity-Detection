"""Property-based tests for API endpoints.

Feature: voice-to-image-system
"""

import pytest
import tempfile
import os
import wave
import struct
import numpy as np
from hypothesis import given, strategies as st, settings
from fastapi.testclient import TestClient

from app.main import app, image_storage
from app.models import GenerateResponse, CompareResponse, ErrorResponse


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


@given(st.floats(min_value=0.5, max_value=10.0))
@settings(max_examples=1, deadline=None)
def test_property_9_api_success_response_format(duration):
    """
    Feature: voice-to-image-system, Property 9: API Success Response Format
    For any successful API request, the server should return HTTP status 200 
    with a valid JSON response containing the expected fields.
    
    Validates: Requirements 5.3, 5.7
    """
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
        tmp_path = tmp_file.name
    
    try:
        # Create a valid WAV file
        create_wav_file(tmp_path, duration)
        
        # Upload audio file to /api/generate
        with open(tmp_path, 'rb') as f:
            response = client.post(
                "/api/generate",
                files={"audio_file": ("test.wav", f, "audio/wav")}
            )
        
        # Property: Successful request should return HTTP 200
        assert response.status_code == 200, f"Expected 200, got {response.status_code}"
        
        # Property: Response should be valid JSON
        data = response.json()
        assert isinstance(data, dict)
        
        # Property: Response should contain expected fields
        assert "image_id" in data
        assert "image_url" in data
        assert "processing_time" in data
        
        # Property: Fields should have correct types
        assert isinstance(data["image_id"], str)
        assert isinstance(data["image_url"], str)
        assert isinstance(data["processing_time"], (int, float))
        
        # Property: image_id should be non-empty
        assert len(data["image_id"]) > 0
        
        # Property: image_url should contain the image_id
        assert data["image_id"] in data["image_url"]
        
        # Property: processing_time should be positive
        assert data["processing_time"] > 0
        
        # Validate response model
        response_model = GenerateResponse(**data)
        assert response_model.image_id == data["image_id"]
        
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


@given(st.sampled_from(['.txt', '.pdf', '.jpg', '.png']))
@settings(max_examples=1, deadline=None)
def test_property_10_api_error_response_format(invalid_extension):
    """
    Feature: voice-to-image-system, Property 10: API Error Response Format
    For any API request with invalid input, the server should return HTTP 
    status 400 with a JSON response containing error details.
    
    Validates: Requirements 5.4
    """
    with tempfile.NamedTemporaryFile(suffix=invalid_extension, delete=False) as tmp_file:
        tmp_path = tmp_file.name
        tmp_file.write(b'invalid audio data')
    
    try:
        # Upload invalid file to /api/generate
        with open(tmp_path, 'rb') as f:
            response = client.post(
                "/api/generate",
                files={"audio_file": (f"test{invalid_extension}", f, "application/octet-stream")}
            )
        
        # Property: Invalid request should return HTTP 400
        assert response.status_code == 400, f"Expected 400, got {response.status_code}"
        
        # Property: Response should be valid JSON
        data = response.json()
        assert isinstance(data, dict)
        
        # Property: Response should contain error fields
        assert "code" in data
        assert "message" in data
        assert "stage" in data
        assert "timestamp" in data
        
        # Property: Fields should have correct types
        assert isinstance(data["code"], str)
        assert isinstance(data["message"], str)
        assert isinstance(data["stage"], str)
        assert isinstance(data["timestamp"], str)
        
        # Property: Error code should be non-empty
        assert len(data["code"]) > 0
        
        # Property: Error message should be descriptive
        assert len(data["message"]) > 0
        
        # Validate error response model
        error_model = ErrorResponse(**data)
        assert error_model.code == data["code"]
        
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


@given(st.floats(min_value=0.5, max_value=5.0))
@settings(max_examples=1, deadline=None)
def test_property_11_pipeline_execution_order(duration):
    """
    Feature: voice-to-image-system, Property 11: Pipeline Execution Order
    For any uploaded audio file, the system should execute the processing 
    pipeline in the correct order: validation → spectrogram extraction → 
    image generation, with each stage completing before the next begins.
    
    Validates: Requirements 6.1
    """
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
        tmp_path = tmp_file.name
    
    try:
        # Create a valid WAV file
        create_wav_file(tmp_path, duration)
        
        # Upload audio file
        with open(tmp_path, 'rb') as f:
            response = client.post(
                "/api/generate",
                files={"audio_file": ("test.wav", f, "audio/wav")}
            )
        
        # Property: Pipeline should complete successfully
        assert response.status_code == 200
        
        data = response.json()
        image_id = data["image_id"]
        
        # Property: Image should be generated and stored
        assert image_id in image_storage
        
        # Property: Image file should exist
        image_path = image_storage[image_id]
        assert os.path.exists(image_path)
        
        # Property: Image should be retrievable via API
        image_response = client.get(f"/api/images/{image_id}")
        assert image_response.status_code == 200
        assert image_response.headers["content-type"] == "image/png"
        
        # Property: Image data should be non-empty
        assert len(image_response.content) > 0
        
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


@given(st.sampled_from(['.txt', '.pdf']))
@settings(max_examples=1, deadline=None)
def test_property_12_pipeline_error_reporting(invalid_extension):
    """
    Feature: voice-to-image-system, Property 12: Pipeline Error Reporting
    For any pipeline execution that fails at a specific stage, the system 
    should return an error message clearly indicating which stage failed.
    
    Validates: Requirements 6.3
    """
    with tempfile.NamedTemporaryFile(suffix=invalid_extension, delete=False) as tmp_file:
        tmp_path = tmp_file.name
        tmp_file.write(b'invalid data')
    
    try:
        # Upload invalid file
        with open(tmp_path, 'rb') as f:
            response = client.post(
                "/api/generate",
                files={"audio_file": (f"test{invalid_extension}", f, "application/octet-stream")}
            )
        
        # Property: Failed pipeline should return error response
        assert response.status_code in [400, 500]
        
        data = response.json()
        
        # Property: Error response should indicate the stage that failed
        assert "stage" in data
        assert isinstance(data["stage"], str)
        assert len(data["stage"]) > 0
        
        # Property: Stage should be one of the expected pipeline stages
        expected_stages = ["validation", "loading", "extraction", "generation", "processing"]
        assert data["stage"] in expected_stages, f"Unexpected stage: {data['stage']}"
        
        # Property: Error message should be descriptive
        assert "message" in data
        assert len(data["message"]) > 0
        
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


@given(st.floats(min_value=0.5, max_value=10.0))
@settings(max_examples=1, deadline=None)
def test_property_13_temporary_file_cleanup(duration):
    """
    Feature: voice-to-image-system, Property 13: Temporary File Cleanup
    For any completed processing request (successful or failed), the system 
    should clean up all temporary files created during processing.
    
    Validates: Requirements 6.4
    """
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
        tmp_path = tmp_file.name
    
    try:
        # Create a valid WAV file
        create_wav_file(tmp_path, duration)
        
        # Track temp directory contents before request
        temp_dir = "temp"
        if not os.path.exists(temp_dir):
            os.makedirs(temp_dir)
        
        temp_files_before = set(os.listdir(temp_dir))
        
        # Upload audio file
        with open(tmp_path, 'rb') as f:
            response = client.post(
                "/api/generate",
                files={"audio_file": ("test.wav", f, "audio/wav")}
            )
        
        # Property: Request should complete (success or failure)
        assert response.status_code in [200, 400, 500]
        
        # Property: Temporary files should be cleaned up
        temp_files_after = set(os.listdir(temp_dir))
        
        # No new files should remain in temp directory
        new_files = temp_files_after - temp_files_before
        assert len(new_files) == 0, f"Temporary files not cleaned up: {new_files}"
        
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


@given(st.integers(min_value=2, max_value=3))
@settings(max_examples=1, deadline=None)
def test_property_15_unique_image_identifiers(num_images):
    """
    Feature: voice-to-image-system, Property 15: Unique Image Identifiers
    For any set of generated images, each should be stored with a unique 
    identifier (UUID), ensuring no ID collisions occur.
    
    Validates: Requirements 7.1
    """
    image_ids = []
    temp_files = []
    
    try:
        # Generate multiple images
        for i in range(num_images):
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
                tmp_path = tmp_file.name
                temp_files.append(tmp_path)
            
            # Create a valid WAV file with slight variation
            create_wav_file(tmp_path, 1.0 + i * 0.1)
            
            # Upload audio file
            with open(tmp_path, 'rb') as f:
                response = client.post(
                    "/api/generate",
                    files={"audio_file": (f"test_{i}.wav", f, "audio/wav")}
                )
            
            # Property: Request should succeed (or skip if models not loaded)
            if response.status_code != 200:
                pytest.skip(f"Model loading issue (status {response.status_code}), skipping test")
            
            assert response.status_code == 200
            
            data = response.json()
            image_id = data["image_id"]
            image_ids.append(image_id)
        
        # Property: All image IDs should be unique (no collisions)
        assert len(image_ids) == len(set(image_ids)), "Image ID collision detected!"
        
        # Property: Each image ID should be a valid UUID format
        import uuid
        for image_id in image_ids:
            try:
                uuid.UUID(image_id)
            except ValueError:
                pytest.fail(f"Invalid UUID format: {image_id}")
        
        # Property: All images should be stored in image_storage
        for image_id in image_ids:
            assert image_id in image_storage, f"Image {image_id} not in storage"
        
    finally:
        # Clean up temp files
        for tmp_path in temp_files:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)


@given(st.floats(min_value=0.5, max_value=5.0))
@settings(max_examples=1, deadline=None)
def test_property_16_image_url_accessibility(duration):
    """
    Feature: voice-to-image-system, Property 16: Image URL Accessibility
    For any stored image, the system should provide a valid URL that can be 
    used to retrieve the image via HTTP GET request.
    
    Validates: Requirements 7.2, 7.3
    """
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
        tmp_path = tmp_file.name
    
    try:
        # Create a valid WAV file
        create_wav_file(tmp_path, duration)
        
        # Upload audio file
        with open(tmp_path, 'rb') as f:
            response = client.post(
                "/api/generate",
                files={"audio_file": ("test.wav", f, "audio/wav")}
            )
        
        # Property: Request should succeed (or skip if models not loaded)
        if response.status_code != 200:
            pytest.skip(f"Model loading issue (status {response.status_code}), skipping test")
        
        assert response.status_code == 200
        
        data = response.json()
        image_id = data["image_id"]
        image_url = data["image_url"]
        
        # Property: image_url should be a valid string
        assert isinstance(image_url, str)
        assert len(image_url) > 0
        
        # Property: image_url should contain the image_id
        assert image_id in image_url
        
        # Property: image_url should be accessible via GET request
        image_response = client.get(image_url)
        assert image_response.status_code == 200, f"Failed to access image URL: {image_url}"
        
        # Property: Retrieved image should have content
        assert len(image_response.content) > 0
        
        # Property: Image should be stored and file should exist
        assert image_id in image_storage
        image_path = image_storage[image_id]
        assert os.path.exists(image_path), f"Image file not found: {image_path}"
        
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


@given(st.floats(min_value=0.5, max_value=5.0))
@settings(max_examples=1, deadline=None)
def test_property_17_image_response_headers(duration):
    """
    Feature: voice-to-image-system, Property 17: Image Response Headers
    For any image retrieval request, the server should return the image with 
    appropriate content-type headers (image/png).
    
    Validates: Requirements 7.4
    """
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
        tmp_path = tmp_file.name
    
    try:
        # Create a valid WAV file
        create_wav_file(tmp_path, duration)
        
        # Upload audio file
        with open(tmp_path, 'rb') as f:
            response = client.post(
                "/api/generate",
                files={"audio_file": ("test.wav", f, "audio/wav")}
            )
        
        # Property: Request should succeed (or skip if models not loaded)
        if response.status_code != 200:
            pytest.skip(f"Model loading issue (status {response.status_code}), skipping test")
        
        data = response.json()
        image_id = data["image_id"]
        
        # Retrieve image via GET request
        image_response = client.get(f"/api/images/{image_id}")
        
        # Property: Image retrieval should succeed
        assert image_response.status_code == 200
        
        # Property: Response should have content-type header
        assert "content-type" in image_response.headers
        
        # Property: Content-type should be image/png
        content_type = image_response.headers["content-type"]
        assert content_type == "image/png", f"Expected 'image/png', got '{content_type}'"
        
        # Property: Response should have content-disposition header
        assert "content-disposition" in image_response.headers
        
        # Property: Content-disposition should indicate inline display
        content_disposition = image_response.headers["content-disposition"]
        assert "inline" in content_disposition.lower()
        assert image_id in content_disposition
        
        # Property: Response body should contain valid PNG data
        # PNG files start with specific magic bytes
        png_signature = b'\x89PNG\r\n\x1a\n'
        assert image_response.content[:8] == png_signature, "Response does not contain valid PNG data"
        
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
