# Web Interface Guide

## Overview

The Voice to Image System now includes a modern, user-friendly web interface for uploading audio files, generating images, and comparing similarity between generated images.

## Features

### 1. Audio Upload
- **Drag & Drop**: Drag audio files directly onto the upload zone
- **File Browser**: Click the upload zone to browse and select files
- **Supported Formats**: WAV, MP3, FLAC, OGG
- **File Validation**: Client-side validation for format and size (max 10MB)
- **Real-time Feedback**: Loading indicators and progress updates

### 2. Image Generation
- **Automatic Processing**: Audio is automatically processed after upload
- **Visual Display**: Generated images are displayed in a responsive grid
- **Metadata Display**: Shows filename, processing time, and image ID
- **Transcription Display**: Shows the transcribed text from the audio (if available)

### 3. Image Comparison
- **Selection**: Check up to 2 images to compare
- **Similarity Score**: Displays percentage similarity (0-100%)
- **Color Coding**: 
  - Red (<30%): Low similarity
  - Yellow (30-70%): Medium similarity
  - Green (>70%): High similarity
- **Side-by-Side View**: Visual comparison of selected images

### 4. Error Handling
- **Toast Notifications**: User-friendly error and success messages
- **Retry Mechanism**: Automatic retry for failed uploads
- **Error Types**: Distinguishes between client and server errors

## Usage

### Starting the Server

1. Ensure all dependencies are installed:
   ```bash
   pip install -r requirements.txt
   ```

2. Start the FastAPI server:
   ```bash
   uvicorn app.main:app --reload
   ```

3. Open your browser and navigate to:
   ```
   http://localhost:8000
   ```

### Uploading Audio

1. **Method 1 - Drag & Drop**:
   - Drag an audio file from your file explorer
   - Drop it onto the upload zone
   - Wait for processing to complete

2. **Method 2 - File Browser**:
   - Click on the upload zone
   - Select an audio file from the file dialog
   - Wait for processing to complete

### Comparing Images

1. Upload at least 2 audio files to generate images
2. Check the checkbox on 2 images you want to compare
3. Click the "Compare Selected" button
4. View the similarity score and side-by-side comparison

## Design Features

### Dark Mode Theme
- Modern dark color scheme for reduced eye strain
- Gradient backgrounds and smooth transitions
- High contrast for accessibility

### Responsive Design
- Mobile-friendly layout
- Adapts to different screen sizes
- Touch-friendly controls

### Visual Feedback
- Loading spinners during processing
- Hover effects on interactive elements
- Smooth animations and transitions
- Toast notifications for user feedback

## Technical Details

### File Structure
```
static/
├── index.html    # Main HTML structure
├── styles.css    # Styling and layout
└── app.js        # Client-side logic
```

### API Integration
The web interface communicates with the following API endpoints:

- `POST /api/generate` - Upload audio and generate image
- `POST /api/compare` - Compare two images
- `GET /api/images/{image_id}` - Retrieve generated image

### Browser Compatibility
- Modern browsers (Chrome, Firefox, Safari, Edge)
- Requires JavaScript enabled
- Supports drag & drop API
- Uses Fetch API for HTTP requests

## Troubleshooting

### Images Not Loading
- Check that the server is running
- Verify the images directory exists
- Check browser console for errors

### Upload Fails
- Verify file format is supported (WAV, MP3, FLAC, OGG)
- Check file size is under 10MB
- Ensure audio duration is under 30 seconds
- Check server logs for detailed error messages

### Comparison Not Working
- Ensure exactly 2 images are selected
- Verify both images were successfully generated
- Check browser console for errors

## Future Enhancements

Potential improvements for the web interface:
- Image gallery with pagination
- Download generated images
- Audio playback preview
- Batch upload support
- Image history and persistence
- Advanced filtering and sorting
- Export comparison results
