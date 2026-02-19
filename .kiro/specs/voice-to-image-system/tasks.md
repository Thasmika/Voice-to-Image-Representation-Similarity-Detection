# Implementation Plan: Voice to Image Representation & Similarity Detection System

## Overview

This implementation plan breaks down the Voice to Image system into discrete coding tasks. The system converts audio to semantic images using speech recognition and generative AI, then performs similarity analysis using CLIP embeddings. The implementation follows a bottom-up approach: core modules first, then API layer, then web interface, with testing integrated throughout.

**Key Technology Stack:**
- **Whisper**: Speech-to-text transcription (OpenAI)
- **Stable Diffusion 2.1**: Text-to-image generation (Stability AI)
- **CLIP**: Image similarity analysis (OpenAI)
- **FastAPI**: REST API backend
- **Librosa**: Audio validation and processing

## Tasks

- [x] 1. Set up project structure and dependencies
  - Create directory structure (app/, static/, tests/, temp/, images/)
  - Create requirements.txt with all dependencies (fastapi, librosa, transformers, torch, etc.)
  - Create .gitignore for temp files and generated images
  - Set up virtual environment instructions in README.md
  - _Requirements: 12.3_

- [x] 2. Implement Audio Processor module
  - [x] 2.1 Create audio_processor.py with AudioData and ValidationResult dataclasses
    - Define AudioData with samples, sample_rate, duration, file_path fields
    - Define ValidationResult with is_valid, error_message, file_format, duration fields
    - _Requirements: 1.1, 1.2, 1.3_
  
  - [x] 2.2 Implement audio file validation function
    - Validate file format (WAV, MP3, FLAC, OGG) using file extension and magic numbers
    - Check file exists and is readable
    - Return ValidationResult with appropriate error messages for unsupported formats
    - _Requirements: 1.1, 1.2_
  
  - [x] 2.3 Implement audio loading function
    - Use librosa.load() to load audio with sample_rate=22050
    - Calculate duration and validate against 30-second limit
    - Return AudioData object or raise exception for duration violations
    - _Requirements: 1.3, 1.4, 1.5_
  
  - [x] 2.4 Implement spectrogram extraction function
    - Extract mel-spectrogram using librosa.feature.melspectrogram (n_fft=2048, hop_length=512, n_mels=128)
    - Convert to log scale using librosa.power_to_db
    - Normalize to [0, 1] range
    - Return Spectrogram dataclass with 2D numpy array
    - _Requirements: 2.1, 2.3, 2.4_
  
  - [x] 2.5 Write property test for audio validation
    - **Property 1: Valid Audio Processing**
    - **Validates: Requirements 1.1, 1.3**
  
  - [x] 2.6 Write property test for invalid format rejection
    - **Property 2: Invalid Format Rejection**
    - **Validates: Requirements 1.2**
  
  - [x] 2.7 Write property test for duration limit enforcement
    - **Property 3: Duration Limit Enforcement**
    - **Validates: Requirements 1.4**
  
  - [x] 2.8 Write property test for spectrogram extraction consistency
    - **Property 4: Spectrogram Extraction Consistency**
    - **Validates: Requirements 2.1, 2.3, 2.4**
  
  - [x] 2.9 Write property test for spectrogram determinism
    - **Property 5: Spectrogram Determinism**
    - **Validates: Requirements 2.5**
  
  - [x] 2.10 Write unit tests for edge cases
    - Test exactly 30-second audio
    - Test empty/silent audio
    - Test corrupted audio files
    - _Requirements: 1.4, 1.5, 2.1_

- [x] 3. Checkpoint - Ensure audio processing tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [x] 4. Implement Image Generator module (Semantic Generation)
  - [x] 4.1 Create image_generator.py with GeneratedImage dataclass
    - Define GeneratedImage with image_id, image_data, file_path, created_at, source_audio, transcribed_text fields
    - _Requirements: 3.1, 3.3_
  
  - [x] 4.2 Implement Whisper speech recognition integration
    - Load Whisper model (base) for speech-to-text transcription
    - Implement transcribe_audio() function to convert audio to text
    - Handle model caching with singleton pattern
    - _Requirements: 3.1, 3.3_
  
  - [x] 4.3 Implement Stable Diffusion image generation
    - Load Stable Diffusion 2.1 model for text-to-image generation
    - Implement generate_image_from_text() function
    - Configure for 512x512 photorealistic output
    - Enable memory optimizations (attention slicing, VAE slicing)
    - _Requirements: 3.1, 3.3_
  
  - [x] 4.4 Implement prompt enhancement function
    - Create enhance_prompt() to add quality modifiers
    - Handle empty/short text with default prompts
    - Add descriptors: "detailed", "high quality", "photorealistic"
    - _Requirements: 3.1_
  
  - [x] 4.5 Implement audio-to-image pipeline
    - Create generate_image_from_audio() combining transcription + generation
    - Integrate Whisper transcription with Stable Diffusion generation
    - Return both image and transcribed text
    - _Requirements: 3.1, 3.3_
  
  - [x] 4.6 Implement image saving function
    - Generate UUID for image identifier
    - Save image as PNG to images/ directory
    - Store transcribed text in metadata
    - Return GeneratedImage object with metadata
    - _Requirements: 3.3, 7.1_
  
  - [x] 4.7 Write property test for image generation success
    - **Property 6: Image Generation Success**
    - Test with various text prompts
    - Verify 512x512 RGB PNG output
    - **Validates: Requirements 3.1, 3.3**
  
  - [x] 4.8 Write property test for image generation determinism
    - **Property 7: Image Generation Determinism**
    - Test with same seed produces identical images
    - **Validates: Requirements 3.4**
  
  - [x] 4.9 Write unit tests for semantic generation
    - Test text-to-image with various prompts
    - Test Whisper transcription
    - Test prompt enhancement
    - Verify PNG format and dimensions
    - Test file saving and retrieval with transcribed text
    - _Requirements: 3.1, 3.3, 3.4_

- [x] 5. Implement Similarity Analyzer module
  - [x] 5.1 Create similarity_analyzer.py with SimilarityResult dataclass
    - Define SimilarityResult with image_id_1, image_id_2, similarity_score, embedding_1, embedding_2 fields
    - _Requirements: 4.1, 4.2_
  
  - [x] 5.2 Implement CLIP model loading
    - Load CLIPModel and CLIPProcessor from transformers (openai/clip-vit-base-patch32)
    - Implement singleton pattern to load model once
    - Handle model loading errors with descriptive messages
    - _Requirements: 4.1, 11.1, 11.4_
  
  - [x] 5.3 Implement CLIP embedding computation function
    - Preprocess image using CLIPProcessor
    - Extract image features using model.get_image_features()
    - Normalize embeddings to unit vectors
    - Return numpy array embedding
    - _Requirements: 4.1_
  
  - [x] 5.4 Implement similarity calculation function
    - Compute cosine similarity between two embeddings
    - Ensure result is in [0.0, 1.0] range
    - Return SimilarityResult with score and cached embeddings
    - _Requirements: 4.2, 4.5_
  
  - [x] 5.5 Write property test for similarity score range
    - **Property 8: Similarity Score Range**
    - **Validates: Requirements 4.1, 4.2**
  
  - [x] 5.6 Write unit tests for similarity analysis
    - Test identical images (should return 1.0)
    - Test very different images (should return low score)
    - Test embedding caching
    - _Requirements: 4.1, 4.2, 4.3_
  
  - [x] 5.7 Write property test for model caching
    - **Property 18: Model Caching**
    - **Validates: Requirements 11.4**

- [x] 6. Checkpoint - Ensure core modules tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [x] 6.5 Upgrade to Semantic Generation (System Refactor)
  - [x] 6.5.1 Remove old AudioLDM/spectrogram approach
    - Delete AUDIOLDM_IMPLEMENTATION.md
    - Delete test_audioldm_quick.py
    - Delete download_models.py
    - Delete MODEL_DOWNLOAD_STATUS.md
    - Remove spectrogram visualization code from image_generator.py
    - _Requirements: 3.1, 3.3_
  
  - [x] 6.5.2 Implement Whisper speech recognition
    - Add openai-whisper to requirements.txt
    - Implement load_whisper_model() function
    - Implement transcribe_audio() function
    - Add model caching with singleton pattern
    - _Requirements: 3.1_
  
  - [x] 6.5.3 Implement Stable Diffusion image generation
    - Add Stable Diffusion 2.1 pipeline
    - Implement load_stable_diffusion_model() function
    - Implement generate_image_from_text() function
    - Configure for 512x512 photorealistic output
    - Enable memory optimizations (attention slicing, VAE slicing)
    - _Requirements: 3.1, 3.3_
  
  - [x] 6.5.4 Implement semantic pipeline integration
    - Create generate_image_from_audio() combining Whisper + Stable Diffusion
    - Implement enhance_prompt() for better image quality
    - Update spectrogram_to_image() to use new pipeline
    - Add transcribed_text field to GeneratedImage dataclass
    - Update save_image() to store transcribed text
    - _Requirements: 3.1, 3.3_
  
  - [x] 6.5.5 Update API endpoints for semantic generation
    - Update /api/generate to use audio path instead of spectrogram
    - Add transcribed_text to GenerateResponse model
    - Update startup to load Whisper + Stable Diffusion + CLIP
    - Update /api/health to report all 3 models
    - Fix JSON serialization for datetime in error responses
    - _Requirements: 5.1, 5.3, 11.1_
  
  - [x] 6.5.6 Update all tests for semantic generation
    - Rewrite test_image_generator_unit.py for text-to-image
    - Rewrite test_image_generator_properties.py for semantic generation
    - Update test_api_unit.py for transcribed_text field
    - Update test_api_properties.py for new pipeline
    - Add tests for Whisper transcription
    - Add tests for prompt enhancement
    - _Requirements: 3.1, 3.3, 3.4_
  
  - [x] 6.5.7 Create new documentation
    - Create SEMANTIC_GENERATION_GUIDE.md
    - Create QUICKSTART_SEMANTIC.md
    - Create IMPLEMENTATION_CHANGES.md
    - Create UPGRADE_COMPLETE.md
    - Update README.md for semantic generation
    - Create test_semantic_generation.py
    - _Requirements: 11.5, 12.3_

- [x] 7. Implement FastAPI server and endpoints
  - [x] 7.1 Create main.py with FastAPI app initialization
    - Initialize FastAPI app with CORS middleware
    - Configure static file serving for web interface
    - Set up startup event to load Whisper, Stable Diffusion, and CLIP models
    - Set up shutdown event for cleanup
    - _Requirements: 5.1, 5.2, 12.1, 12.2_
  
  - [x] 7.2 Create models.py with Pydantic request/response models
    - Define GenerateResponse model (image_id, image_url, processing_time, transcribed_text)
    - Define CompareRequest model (image_id_1, image_id_2)
    - Define CompareResponse model (similarity_score, percentage)
    - Define ErrorResponse model (code, message, stage, timestamp)
    - _Requirements: 5.3, 5.4, 5.7_
  
  - [x] 7.3 Implement POST /api/generate endpoint
    - Accept multipart/form-data with audio file
    - Save uploaded file to temp directory
    - Execute pipeline: validate → load → transcribe → generate semantic image
    - Clean up temporary audio file
    - Return GenerateResponse with image URL and transcribed text
    - _Requirements: 5.1, 5.3, 5.6, 6.1_
  
  - [x] 7.4 Implement POST /api/compare endpoint
    - Accept CompareRequest with two image IDs
    - Validate both image IDs exist
    - Load images from storage
    - Compute embeddings and similarity score
    - Return CompareResponse with score and percentage
    - _Requirements: 5.2, 5.3, 5.7_
  
  - [x] 7.5 Implement GET /api/images/{image_id} endpoint
    - Validate image_id exists
    - Serve image file with content-type: image/png
    - Return 404 if image not found
    - _Requirements: 7.3, 7.4_
  
  - [x] 7.6 Implement GET /api/health endpoint
    - Check if Whisper, Stable Diffusion, and CLIP models are loaded
    - Return status and individual model loading flags
    - _Requirements: 11.1_
  
  - [x] 7.7 Implement custom exception handlers
    - Handle validation errors (400 responses)
    - Handle processing errors (500 responses)
    - Handle file not found errors (404 responses)
    - Return consistent ErrorResponse format with JSON serialization
    - _Requirements: 5.4, 5.5, 10.4_
  
  - [x] 7.8 Write property test for API success responses
    - **Property 9: API Success Response Format**
    - Test with various audio durations
    - Verify response includes transcribed_text
    - **Validates: Requirements 5.3, 5.7**
  
  - [x] 7.9 Write property test for API error responses
    - **Property 10: API Error Response Format**
    - Test with invalid file formats
    - **Validates: Requirements 5.4**
  
  - [x] 7.10 Write property test for pipeline execution order
    - **Property 11: Pipeline Execution Order**
    - Verify: validation → transcription → generation pipeline
    - **Validates: Requirements 6.1**
  
  - [x] 7.11 Write property test for pipeline error reporting
    - **Property 12: Pipeline Error Reporting**
    - Test error reporting at each stage
    - **Validates: Requirements 6.3**
  
  - [x] 7.12 Write unit tests for each endpoint
    - Test /api/generate with valid audio (returns transcribed_text)
    - Test /api/generate with invalid audio
    - Test /api/compare with valid image IDs
    - Test /api/compare with invalid image IDs
    - Test /api/images/{image_id} retrieval
    - Test /api/health endpoint (checks all 3 models)
    - _Requirements: 5.1, 5.2, 5.3, 5.4, 7.3, 7.4_

- [x] 8. Implement resource management and cleanup
  - [x] 8.1 Implement temporary file cleanup function
    - Clean up temp audio files after processing
    - Use try-finally blocks to ensure cleanup
    - Log cleanup operations
    - _Requirements: 6.4_
  
  - [x] 8.2 Implement image storage management
    - Create images directory if not exists
    - Store images with UUID filenames
    - Implement in-memory dict for image metadata
    - _Requirements: 7.1, 7.2_
  
  - [x] 8.3 Write property test for temporary file cleanup
    - **Property 13: Temporary File Cleanup**
    - **Validates: Requirements 6.4**
  
  - [x] 8.4 Write property test for unique image identifiers
    - **Property 15: Unique Image Identifiers**
    - **Validates: Requirements 7.1**
  
  - [x] 8.5 Write property test for image URL accessibility
    - **Property 16: Image URL Accessibility**
    - **Validates: Requirements 7.2, 7.3**
  
  - [x] 8.6 Write property test for image response headers
    - **Property 17: Image Response Headers**
    - **Validates: Requirements 7.4**

- [ ] 9. Checkpoint - Ensure API tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [x] 10. Implement web interface
  - [x] 10.1 Create static/index.html with page structure
    - Create header with title "Voice to Image System"
    - Create upload section with drag-and-drop zone
    - Create results section for displaying generated images
    - Create comparison section for side-by-side display
    - Use semantic HTML5 elements
    - _Requirements: 8.1, 8.2, 8.4, 9.1, 9.2_
  
  - [x] 10.2 Create static/styles.css with modern styling
    - Implement responsive grid layout
    - Style upload zone with hover effects
    - Style image cards with shadows and borders
    - Style similarity score display with color coding (red/yellow/green)
    - Implement dark mode color scheme
    - Add loading spinner animations
    - _Requirements: 9.1, 9.3, 9.4, 9.5_
  
  - [x] 10.3 Create static/app.js with upload functionality
    - Implement drag-and-drop file handling
    - Implement file input button fallback
    - Validate file format client-side
    - Display filename and file size on selection
    - Show loading indicator during upload
    - _Requirements: 8.1, 8.3, 8.5_
  
  - [x] 10.4 Implement image generation API call in app.js
    - Send POST request to /api/generate with FormData
    - Handle response and display generated image
    - Display processing time
    - Add image to results grid with checkbox
    - Handle errors and display user-friendly messages
    - _Requirements: 8.6, 9.1, 10.1_
  
  - [x] 10.5 Implement image comparison functionality in app.js
    - Track selected images via checkboxes
    - Enable "Compare Selected" button when 2 images selected
    - Send POST request to /api/compare
    - Display similarity score as percentage
    - Display progress bar with color coding
    - Show images side-by-side
    - _Requirements: 9.2, 9.3, 9.4_
  
  - [x] 10.6 Implement error handling and user feedback in app.js
    - Display toast notifications for errors
    - Distinguish between client and server errors
    - Allow retry without page refresh
    - Display success confirmations
    - _Requirements: 10.1, 10.2, 10.3, 10.5_

- [-] 11. Implement concurrent request handling
  - [x] 11.1 Add async/await to all API endpoints
    - Make file I/O operations async
    - Use asyncio for concurrent processing where applicable
    - _Requirements: 6.5_
  
  - [x] 11.2 Write property test for concurrent request safety
    - **Property 14: Concurrent Request Safety**
    - **Validates: Requirements 6.5**
  
  - [ ] 11.3 Write integration tests for concurrent requests
    - Test multiple simultaneous uploads
    - Test multiple simultaneous comparisons
    - Verify no data corruption
    - _Requirements: 6.5_

- [x] 12. Create configuration and documentation
  - [x] 12.1 Create config.py with configuration settings
    - Define constants for sample rate, spectrogram parameters
    - Define paths for temp and images directories
    - Define server host and port (localhost:8000)
    - Make port configurable via environment variable
    - _Requirements: 12.1, 12.2_
  
  - [x] 12.2 Update README.md with setup and usage instructions
    - Document system requirements
    - Document installation steps (virtual environment, dependencies)
    - Document how to run the server
    - Document API endpoints
    - Document pretrained models used (CLIP)
    - _Requirements: 11.5, 12.3_
  
  - [x] 12.3 Add startup logging
    - Log server URL on startup
    - Log CLIP model loading status
    - Log any initialization errors
    - _Requirements: 12.4_

- [ ] 13. Final integration and testing
  - [ ] 13.1 Write end-to-end integration tests
    - Test complete flow: upload → generate → compare
    - Test multiple audio files in sequence
    - Test error recovery
    - _Requirements: 6.1, 6.2, 6.3_
  
  - [ ] 13.2 Run all tests and verify coverage
    - Run pytest with coverage report
    - Ensure >80% code coverage
    - Fix any failing tests
    - _Requirements: All_
  
  - [ ] 13.3 Manual testing checklist
    - Test with various audio formats (WAV, MP3, FLAC, OGG)
    - Test with audio of different durations
    - Test with multiple concurrent uploads
    - Test comparison with different image pairs
    - Test error scenarios (invalid files, missing files)
    - Verify web interface responsiveness
    - _Requirements: All_

- [ ] 14. Final checkpoint - System ready for use
  - Ensure all tests pass, ask the user if questions arise.

## Notes

- Tasks marked with `*` are optional and can be skipped for faster MVP
- Each task references specific requirements for traceability
- Property tests validate universal correctness properties across all inputs
- Unit tests validate specific examples and edge cases
- The implementation uses Python with FastAPI, Whisper, Stable Diffusion, CLIP, and standard web technologies
- All pretrained models are loaded at startup and cached for efficiency
- The system is designed for localhost deployment with potential for future scaling
- **Semantic Generation**: The system now generates photorealistic images based on speech content, not abstract spectrograms
- **Model Downloads**: First run requires ~6GB of model downloads (Whisper ~140MB, Stable Diffusion ~5GB, CLIP ~350MB)
- **Performance**: GPU recommended for practical use (5-8 seconds per audio), CPU mode is slow (35-65 seconds per audio)
