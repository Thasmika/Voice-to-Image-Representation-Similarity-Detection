"""Unit tests for audio processor edge cases.

Feature: voice-to-image-system
"""

import pytest
import numpy as np
import tempfile
import os
import wave
import struct

from app.audio_processor import (
    validate_audio, load_audio, extract_spectrogram,
    AudioData, ValidationResult, Spectrogram
)


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


def test_exactly_30_second_audio():
    """Test audio file with exactly 30 seconds duration.
    
    Requirements: 1.4
    """
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
        tmp_path = tmp_file.name
    
    try:
        # Create exactly 30-second audio
        create_wav_file(tmp_path, 30.0)
        
        # Should load successfully
        audio_data = load_audio(tmp_path)
        assert audio_data.duration <= 30.0
        assert audio_data.duration >= 29.9  # Allow small tolerance
        
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


def test_empty_silent_audio():
    """Test audio file with silent/zero amplitude audio.
    
    Requirements: 1.5, 2.1
    """
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
        tmp_path = tmp_file.name
    
    try:
        # Create silent audio (all zeros)
        duration = 2.0
        sample_rate = 22050
        num_samples = int(duration * sample_rate)
        
        with wave.open(tmp_path, 'w') as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(sample_rate)
            
            for _ in range(num_samples):
                data = struct.pack('<h', 0)  # Silent
                wav_file.writeframes(data)
        
        # Should load successfully
        audio_data = load_audio(tmp_path)
        assert audio_data.samples is not None
        assert len(audio_data.samples) > 0
        
        # Should extract spectrogram successfully
        spectrogram = extract_spectrogram(audio_data)
        assert spectrogram.data is not None
        assert spectrogram.data.shape[0] == 128
        
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


def test_corrupted_audio_file():
    """Test handling of corrupted audio files.
    
    Requirements: 1.5
    """
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
        tmp_path = tmp_file.name
        # Write corrupted/invalid WAV data
        tmp_file.write(b'RIFF' + b'\x00' * 100)  # Invalid WAV structure
    
    try:
        # Validation should fail or return error
        result = validate_audio(tmp_path)
        
        # Either validation fails or loading fails
        if result.is_valid:
            # If validation passes, loading should fail
            with pytest.raises(Exception):
                load_audio(tmp_path)
        else:
            # Validation correctly identified the issue
            assert result.error_message is not None
        
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


def test_very_short_audio():
    """Test audio file with very short duration (< 0.1 seconds).
    
    Requirements: 1.4, 2.1
    """
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
        tmp_path = tmp_file.name
    
    try:
        # Create very short audio (0.05 seconds)
        create_wav_file(tmp_path, 0.05)
        
        # Should load successfully
        audio_data = load_audio(tmp_path)
        assert audio_data.duration < 0.1
        
        # Should extract spectrogram successfully
        spectrogram = extract_spectrogram(audio_data)
        assert spectrogram.data is not None
        
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


def test_nonexistent_file():
    """Test handling of nonexistent file.
    
    Requirements: 1.2
    """
    result = validate_audio('/nonexistent/path/to/audio.wav')
    assert not result.is_valid
    assert "not found" in result.error_message.lower()


def test_unreadable_file():
    """Test handling of file without read permissions.
    
    Requirements: 1.2
    """
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
        tmp_path = tmp_file.name
        create_wav_file(tmp_path, 1.0)
    
    try:
        # Make file unreadable (Windows doesn't support chmod the same way)
        # This test may not work on Windows, so we'll skip it
        import platform
        if platform.system() != 'Windows':
            os.chmod(tmp_path, 0o000)
            result = validate_audio(tmp_path)
            assert not result.is_valid
            assert "not readable" in result.error_message.lower()
        
    finally:
        # Restore permissions and delete
        if platform.system() != 'Windows':
            os.chmod(tmp_path, 0o644)
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
