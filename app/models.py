"""Pydantic models for API request/response validation.

This module defines the data models for FastAPI endpoints.
"""

from pydantic import BaseModel, Field
from datetime import datetime
from typing import Optional


class GenerateResponse(BaseModel):
    """Response model for image generation endpoint."""
    image_id: str = Field(..., description="Unique identifier for the generated image")
    image_url: str = Field(..., description="URL to access the generated image")
    processing_time: float = Field(..., description="Time taken to process in seconds")
    transcribed_text: Optional[str] = Field(None, description="Transcribed text from audio (if available)")


class CompareRequest(BaseModel):
    """Request model for image comparison endpoint."""
    image_id_1: str = Field(..., description="First image identifier")
    image_id_2: str = Field(..., description="Second image identifier")


class CompareResponse(BaseModel):
    """Response model for image comparison endpoint."""
    similarity_score: float = Field(..., description="Similarity score between 0.0 and 1.0")
    percentage: str = Field(..., description="Similarity as percentage string (e.g., '87%')")


class ErrorResponse(BaseModel):
    """Response model for error responses."""
    code: str = Field(..., description="Error code identifier")
    message: str = Field(..., description="Human-readable error message")
    stage: str = Field(..., description="Processing stage where error occurred")
    timestamp: datetime = Field(default_factory=datetime.now, description="Error timestamp")
