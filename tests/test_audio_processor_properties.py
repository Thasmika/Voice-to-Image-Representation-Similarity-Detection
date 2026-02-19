"""Property-based tests for audio processor module.

Feature: voice-to-image-system
"""

import pytest
import numpy as np
import tempfile
import os
from hypothesis import given, strategies as st, settings
import wave
import struct

from app.audio_processor import (
    validate_audio, load_audio, extract_spectrogram,
    AudioData, ValidationResult, Spectrogram
)


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


@given(st.floats(min_value=0.1, max_value=30.0))
@settings(max_examples=3, deadline=None)
def test_property_1_valid_audio_processing(duration):
    """
    Feature: voice-to-image-system, Property 1: Valid Audio Processing
    For any valid audio file in supported formats (WAV, MP3, FLAC, OGG), 
    validation should succeed and the audio should load without errors.
    
    Validates: Requirements 1.1, 1.3
    """
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
        tmp_path = tmp_file.name
    
    try:
        # Create a valid WAV file
        create_wav_file(tmp_path, duration)
        
        # Validate the audio file
        result = validate_audio(tmp_path)
        
        # Property: Validation should succeed for valid audio files
        assert result.is_valid, f"Valid audio file failed validation: {result.error_message}"
        assert result.error_message is None
        assert result.file_format == '.wav'
        assert result.duration is not None
        assert result.duration > 0
        
        # Property: Valid audio should load successfully
        audio_data = load_audio(tmp_path)
        assert isinstance(audio_data, AudioData)
        assert audio_data.samples is not None
        assert len(audio_data.samples) > 0
        assert audio_data.sample_rate == 22050
        assert audio_data.duration > 0
        
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)



@given(st.sampled_from(['.txt', '.pdf', '.jpg', '.png', '.aac', '.m4a', '.wma']))
@settings(max_examples=3, deadline=None)
def test_property_2_invalid_format_rejection(invalid_extension):
    """
    Feature: voice-to-image-system, Property 2: Invalid Format Rejection
    For any file with an unsupported audio format, the system should return 
    a descriptive error message indicating the format is not supported.
    
    Validates: Requirements 1.2
    """
    with tempfile.NamedTemporaryFile(suffix=invalid_extension, delete=False) as tmp_file:
        tmp_path = tmp_file.name
        # Write some dummy data
        tmp_file.write(b'dummy data')
    
    try:
        # Validate the file with unsupported format
        result = validate_audio(tmp_path)
        
        # Property: Validation should fail for unsupported formats
        assert not result.is_valid, f"Unsupported format {invalid_extension} was incorrectly validated as valid"
        assert result.error_message is not None
        assert "Unsupported audio format" in result.error_message or "not match expected format" in result.error_message
        
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)



@given(st.floats(min_value=30.1, max_value=60.0))
@settings(max_examples=3, deadline=None)
def test_property_3_duration_limit_enforcement(duration):
    """
    Feature: voice-to-image-system, Property 3: Duration Limit Enforcement
    For any audio file with duration up to 30 seconds, the system should process 
    it successfully; for any audio exceeding 30 seconds, the system should reject 
    it with an appropriate error.
    
    Validates: Requirements 1.4
    """
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
        tmp_path = tmp_file.name
    
    try:
        # Create a WAV file exceeding the duration limit
        create_wav_file(tmp_path, duration)
        
        # Property: Audio exceeding 30 seconds should be rejected
        with pytest.raises(ValueError) as exc_info:
            load_audio(tmp_path)
        
        assert "exceeds maximum limit" in str(exc_info.value)
        
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


@given(st.floats(min_value=0.1, max_value=30.0))
@settings(max_examples=3, deadline=None)
def test_property_3_duration_limit_enforcement_valid(duration):
    """
    Feature: voice-to-image-system, Property 3: Duration Limit Enforcement (valid case)
    For any audio file with duration up to 30 seconds, the system should process 
    it successfully.
    
    Validates: Requirements 1.4
    """
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
        tmp_path = tmp_file.name
    
    try:
        # Create a WAV file within the duration limit
        create_wav_file(tmp_path, duration)
        
        # Property: Audio within 30 seconds should load successfully
        audio_data = load_audio(tmp_path)
        assert audio_data.duration <= 30.0
        
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)



@given(st.floats(min_value=0.5, max_value=10.0))
@settings(max_examples=3, deadline=None)
def test_property_4_spectrogram_extraction_consistency(duration):
    """
    Feature: voice-to-image-system, Property 4: Spectrogram Extraction Consistency
    For any valid audio data, extracting a mel-spectrogram should produce a 2D 
    numpy array with normalized values in a consistent scale.
    
    Validates: Requirements 2.1, 2.3, 2.4
    """
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
        tmp_path = tmp_file.name
    
    try:
        # Create a valid WAV file
        create_wav_file(tmp_path, duration)
        
        # Load audio
        audio_data = load_audio(tmp_path)
        
        # Extract spectrogram
        spectrogram = extract_spectrogram(audio_data)
        
        # Property: Spectrogram should be a 2D numpy array
        assert isinstance(spectrogram.data, np.ndarray)
        assert spectrogram.data.ndim == 2
        
        # Property: Values should be normalized to [0, 1] range
        assert np.all(spectrogram.data >= 0.0)
        assert np.all(spectrogram.data <= 1.0)
        
        # Property: Spectrogram should have expected dimensions
        assert spectrogram.n_mels == 128
        assert spectrogram.hop_length == 512
        assert spectrogram.sample_rate == 22050
        
        # Property: Spectrogram should have mel bands as first dimension
        assert spectrogram.data.shape[0] == 128
        
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)



@given(st.floats(min_value=0.5, max_value=10.0))
@settings(max_examples=3, deadline=None)
def test_property_5_spectrogram_determinism(duration):
    """
    Feature: voice-to-image-system, Property 5: Spectrogram Determinism
    For any valid audio input, extracting the spectrogram multiple times 
    should produce identical outputs (deterministic processing).
    
    Validates: Requirements 2.5
    """
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
        tmp_path = tmp_file.name
    
    try:
        # Create a valid WAV file
        create_wav_file(tmp_path, duration)
        
        # Load audio
        audio_data = load_audio(tmp_path)
        
        # Extract spectrogram twice
        spectrogram1 = extract_spectrogram(audio_data)
        spectrogram2 = extract_spectrogram(audio_data)
        
        # Property: Both spectrograms should be identical
        assert np.array_equal(spectrogram1.data, spectrogram2.data), \
            "Spectrogram extraction is not deterministic"
        
        # Property: Metadata should also be identical
        assert spectrogram1.sample_rate == spectrogram2.sample_rate
        assert spectrogram1.hop_length == spectrogram2.hop_length
        assert spectrogram1.n_mels == spectrogram2.n_mels
        
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
