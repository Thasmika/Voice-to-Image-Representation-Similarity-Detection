"""Audio processing module for voice-to-image system.

This module handles audio file validation, loading, and spectrogram extraction.
"""

from dataclasses import dataclass
from typing import Optional
import numpy as np
import librosa
import os


@dataclass
class AudioData:
    """Container for loaded audio data."""
    samples: np.ndarray
    sample_rate: int
    duration: float
    file_path: str


@dataclass
class ValidationResult:
    """Result of audio file validation."""
    is_valid: bool
    error_message: Optional[str]
    file_format: Optional[str]
    duration: Optional[float]


@dataclass
class Spectrogram:
    """Container for spectrogram data."""
    data: np.ndarray
    sample_rate: int
    hop_length: int
    n_mels: int



# Supported audio formats
SUPPORTED_FORMATS = {'.wav', '.mp3', '.flac', '.ogg'}

# Magic numbers for file format validation
MAGIC_NUMBERS = {
    b'RIFF': 'wav',
    b'ID3': 'mp3',
    b'\xff\xfb': 'mp3',
    b'\xff\xf3': 'mp3',
    b'\xff\xf2': 'mp3',
    b'fLaC': 'flac',
    b'OggS': 'ogg',
}


def validate_audio(file_path: str) -> ValidationResult:
    """Validate audio file format and readability.
    
    Args:
        file_path: Path to the audio file
        
    Returns:
        ValidationResult with validation status and details
    """
    # Check if file exists
    if not os.path.exists(file_path):
        return ValidationResult(
            is_valid=False,
            error_message=f"File not found: {file_path}",
            file_format=None,
            duration=None
        )
    
    # Check if file is readable
    if not os.access(file_path, os.R_OK):
        return ValidationResult(
            is_valid=False,
            error_message=f"File is not readable: {file_path}",
            file_format=None,
            duration=None
        )
    
    # Validate file extension
    _, ext = os.path.splitext(file_path.lower())
    if ext not in SUPPORTED_FORMATS:
        return ValidationResult(
            is_valid=False,
            error_message=f"Unsupported audio format: {ext}. Supported formats: WAV, MP3, FLAC, OGG",
            file_format=ext,
            duration=None
        )
    
    # Validate using magic numbers
    try:
        with open(file_path, 'rb') as f:
            header = f.read(4)
            
        format_detected = None
        for magic, fmt in MAGIC_NUMBERS.items():
            if header.startswith(magic):
                format_detected = fmt
                break
        
        if format_detected is None:
            return ValidationResult(
                is_valid=False,
                error_message=f"File header does not match expected format for {ext}",
                file_format=ext,
                duration=None
            )
        
        # Try to get duration using librosa
        try:
            duration = librosa.get_duration(path=file_path)
        except Exception as e:
            return ValidationResult(
                is_valid=False,
                error_message=f"Failed to read audio file: {str(e)}",
                file_format=ext,
                duration=None
            )
        
        return ValidationResult(
            is_valid=True,
            error_message=None,
            file_format=ext,
            duration=duration
        )
        
    except Exception as e:
        return ValidationResult(
            is_valid=False,
            error_message=f"Error validating file: {str(e)}",
            file_format=ext,
            duration=None
        )



# Maximum audio duration in seconds
MAX_DURATION = 30.0


def load_audio(file_path: str, sample_rate: int = 22050) -> AudioData:
    """Load audio file and return AudioData object.
    
    Args:
        file_path: Path to the audio file
        sample_rate: Target sample rate (default: 22050 Hz)
        
    Returns:
        AudioData object with loaded audio
        
    Raises:
        ValueError: If audio duration exceeds maximum limit
        Exception: If audio loading fails
    """
    try:
        # Load audio using librosa
        samples, sr = librosa.load(file_path, sr=sample_rate)
        
        # Calculate duration
        duration = len(samples) / sr
        
        # Validate duration against limit
        if duration > MAX_DURATION:
            raise ValueError(
                f"Audio duration ({duration:.2f}s) exceeds maximum limit of {MAX_DURATION}s"
            )
        
        return AudioData(
            samples=samples,
            sample_rate=sr,
            duration=duration,
            file_path=file_path
        )
        
    except ValueError:
        # Re-raise duration validation errors
        raise
    except Exception as e:
        raise Exception(f"Failed to load audio file: {str(e)}")



def extract_spectrogram(audio_data: AudioData, n_fft: int = 2048, 
                       hop_length: int = 512, n_mels: int = 128) -> Spectrogram:
    """Extract mel-spectrogram from audio data.
    
    Args:
        audio_data: AudioData object containing audio samples
        n_fft: FFT window size (default: 2048)
        hop_length: Number of samples between successive frames (default: 512)
        n_mels: Number of mel bands (default: 128)
        
    Returns:
        Spectrogram object with normalized 2D array
    """
    # Extract mel-spectrogram
    mel_spec = librosa.feature.melspectrogram(
        y=audio_data.samples,
        sr=audio_data.sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels
    )
    
    # Convert to log scale
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    
    # Normalize to [0, 1] range
    min_val = mel_spec_db.min()
    max_val = mel_spec_db.max()
    
    # Handle edge case where all values are the same (e.g., silent audio)
    if max_val - min_val < 1e-10:
        mel_spec_normalized = np.zeros_like(mel_spec_db)
    else:
        mel_spec_normalized = (mel_spec_db - min_val) / (max_val - min_val)
    
    return Spectrogram(
        data=mel_spec_normalized,
        sample_rate=audio_data.sample_rate,
        hop_length=hop_length,
        n_mels=n_mels
    )
