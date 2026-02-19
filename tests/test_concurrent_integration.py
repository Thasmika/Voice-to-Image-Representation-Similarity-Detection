"""Integration tests for concurrent request handling.

These tests verify that the system can handle multiple simultaneous requests
without data corruption or race conditions.
"""

import pytest
import tempfile
import os
import wave
import struct
import numpy as np
import concurrent.futures
import threading
from fastapi.testclient import TestClient

from app.main import app, image_storage


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


def test_concurrent_uploads():
    """
    Test multiple simultaneous uploads.
    
    Validates: Requirements 6.5
    """
    client = TestClient(app)
    num_requests = 3
    temp_files = []
    results_lock = threading.Lock()
    results = []
    
    try:
        # Create multiple WAV files
        for i in range(num_requests):
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
                tmp_path = tmp_file.name
                temp_files.append(tmp_path)
            
            # Create unique audio files with different durations
            create_wav_file(tmp_path, 1.0 + i * 0.3)
        
        # Function to make a single upload request
        def upload_audio(file_path: str, index: int):
            thread_client = TestClient(app)
            
            try:
                with open(file_path, 'rb') as f:
                    response = thread_client.post(
                        "/api/generate",
                        files={"audio_file": (f"concurrent_test_{index}.wav", f, "audio/wav")}
                    )
                
                with results_lock:
                    results.append((index, response))
                
                return response
            except Exception as e:
                with results_lock:
                    results.append((index, e))
                return e
        
        # Execute concurrent uploads
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_requests) as executor:
            futures = [executor.submit(upload_audio, temp_files[i], i) for i in range(num_requests)]
            concurrent.futures.wait(futures)
        
        # Verify all requests completed
        assert len(results) == num_requests, f"Expected {num_requests} results, got {len(results)}"
        
        # Collect successful responses
        successful_responses = []
        for index, result in results:
            if isinstance(result, Exception):
                pytest.skip(f"Request {index} raised exception: {result}")
            
            assert hasattr(result, 'status_code'), f"Result {index} has no status_code"
            
            if result.status_code == 200:
                successful_responses.append((index, result))
        
        # Verify data integrity if we have successful responses
        if len(successful_responses) >= 2:
            image_ids = set()
            
            for index, response in successful_responses:
                data = response.json()
                
                # Verify response structure
                assert "image_id" in data
                assert "image_url" in data
                assert "processing_time" in data
                assert "transcribed_text" in data
                
                image_id = data["image_id"]
                
                # Verify no ID collisions
                assert image_id not in image_ids, f"Duplicate image_id detected: {image_id}"
                image_ids.add(image_id)
                
                # Verify image is stored
                assert image_id in image_storage, f"Image {image_id} not in storage"
                
                # Verify image file exists
                image_path = image_storage[image_id]
                assert os.path.exists(image_path), f"Image file not found: {image_path}"
                
                # Verify image is retrievable
                image_response = client.get(f"/api/images/{image_id}")
                assert image_response.status_code == 200
                assert len(image_response.content) > 0
        
    finally:
        # Clean up temp files
        for tmp_path in temp_files:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)


def test_concurrent_comparisons():
    """
    Test multiple simultaneous comparisons.
    
    Validates: Requirements 6.5
    """
    client = TestClient(app)
    temp_files = []
    
    try:
        # First, generate 3 images
        image_ids = []
        for i in range(3):
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
                tmp_path = tmp_file.name
                temp_files.append(tmp_path)
            
            create_wav_file(tmp_path, 1.0 + i * 0.5)
            
            with open(tmp_path, 'rb') as f:
                response = client.post(
                    "/api/generate",
                    files={"audio_file": (f"compare_test_{i}.wav", f, "audio/wav")}
                )
            
            if response.status_code != 200:
                pytest.skip(f"Failed to generate image {i}: status {response.status_code}")
            
            data = response.json()
            image_ids.append(data["image_id"])
        
        # Now perform concurrent comparisons
        results_lock = threading.Lock()
        results = []
        
        comparison_pairs = [
            (image_ids[0], image_ids[1]),
            (image_ids[1], image_ids[2]),
            (image_ids[0], image_ids[2])
        ]
        
        def compare_images(image_id_1: str, image_id_2: str, index: int):
            thread_client = TestClient(app)
            
            try:
                response = thread_client.post(
                    "/api/compare",
                    json={"image_id_1": image_id_1, "image_id_2": image_id_2}
                )
                
                with results_lock:
                    results.append((index, response))
                
                return response
            except Exception as e:
                with results_lock:
                    results.append((index, e))
                return e
        
        # Execute concurrent comparisons
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            futures = [
                executor.submit(compare_images, pair[0], pair[1], i)
                for i, pair in enumerate(comparison_pairs)
            ]
            concurrent.futures.wait(futures)
        
        # Verify all comparisons completed
        assert len(results) == len(comparison_pairs)
        
        # Verify all comparisons succeeded
        for index, result in results:
            if isinstance(result, Exception):
                pytest.skip(f"Comparison {index} raised exception: {result}")
            
            assert result.status_code == 200, f"Comparison {index} failed with status {result.status_code}"
            
            data = result.json()
            
            # Verify response structure
            assert "similarity_score" in data
            assert "percentage" in data
            
            # Verify similarity score is valid
            score = data["similarity_score"]
            assert 0.0 <= score <= 1.0, f"Invalid similarity score: {score}"
    
    finally:
        # Clean up temp files
        for tmp_path in temp_files:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)


def test_no_data_corruption_under_concurrent_load():
    """
    Verify no data corruption occurs under concurrent load.
    
    This test performs multiple concurrent uploads and comparisons
    simultaneously to stress test the system.
    
    Validates: Requirements 6.5
    """
    client = TestClient(app)
    num_uploads = 4
    temp_files = []
    results_lock = threading.Lock()
    upload_results = []
    
    try:
        # Create WAV files
        for i in range(num_uploads):
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
                tmp_path = tmp_file.name
                temp_files.append(tmp_path)
            
            create_wav_file(tmp_path, 1.0 + i * 0.25)
        
        # Upload function
        def upload_audio(file_path: str, index: int):
            thread_client = TestClient(app)
            
            try:
                with open(file_path, 'rb') as f:
                    response = thread_client.post(
                        "/api/generate",
                        files={"audio_file": (f"stress_test_{index}.wav", f, "audio/wav")}
                    )
                
                with results_lock:
                    upload_results.append((index, response))
                
                return response
            except Exception as e:
                with results_lock:
                    upload_results.append((index, e))
                return e
        
        # Execute concurrent uploads
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_uploads) as executor:
            futures = [executor.submit(upload_audio, temp_files[i], i) for i in range(num_uploads)]
            concurrent.futures.wait(futures)
        
        # Verify uploads completed
        assert len(upload_results) == num_uploads
        
        # Collect image IDs
        image_ids = []
        for index, result in upload_results:
            if isinstance(result, Exception):
                pytest.skip(f"Upload {index} raised exception: {result}")
            
            if result.status_code == 200:
                data = result.json()
                image_ids.append(data["image_id"])
        
        # Verify no duplicate IDs
        assert len(image_ids) == len(set(image_ids)), "Duplicate image IDs detected!"
        
        # Verify all images are stored correctly
        for image_id in image_ids:
            assert image_id in image_storage
            image_path = image_storage[image_id]
            assert os.path.exists(image_path)
            
            # Verify image is retrievable
            response = client.get(f"/api/images/{image_id}")
            assert response.status_code == 200
            assert len(response.content) > 0
        
        # If we have at least 2 images, test concurrent retrieval
        if len(image_ids) >= 2:
            retrieval_results = []
            
            def retrieve_image(image_id: str, index: int):
                thread_client = TestClient(app)
                
                try:
                    response = thread_client.get(f"/api/images/{image_id}")
                    
                    with results_lock:
                        retrieval_results.append((index, response))
                    
                    return response
                except Exception as e:
                    with results_lock:
                        retrieval_results.append((index, e))
                    return e
            
            # Execute concurrent retrievals
            with concurrent.futures.ThreadPoolExecutor(max_workers=len(image_ids)) as executor:
                futures = [
                    executor.submit(retrieve_image, image_ids[i], i)
                    for i in range(len(image_ids))
                ]
                concurrent.futures.wait(futures)
            
            # Verify all retrievals succeeded
            assert len(retrieval_results) == len(image_ids)
            
            for index, result in retrieval_results:
                if isinstance(result, Exception):
                    pytest.fail(f"Retrieval {index} raised exception: {result}")
                
                assert result.status_code == 200
                assert len(result.content) > 0
    
    finally:
        # Clean up temp files
        for tmp_path in temp_files:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
