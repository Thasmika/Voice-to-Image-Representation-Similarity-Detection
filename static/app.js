// Global state
const state = {
    selectedImages: new Set(),
    generatedImages: []
};

// DOM Elements
const dropZone = document.getElementById('drop-zone');
const fileInput = document.getElementById('file-input');
const fileInfo = document.getElementById('file-info');
const fileName = document.getElementById('file-name');
const fileSize = document.getElementById('file-size');
const loadingIndicator = document.getElementById('loading-indicator');
const resultsGrid = document.getElementById('results-grid');
const compareBtn = document.getElementById('compare-btn');
const comparisonSection = document.getElementById('comparison-section');

// Supported audio formats
const SUPPORTED_FORMATS = ['.wav', '.mp3', '.flac', '.ogg'];
const MAX_FILE_SIZE = 10 * 1024 * 1024; // 10MB

// Initialize event listeners
function init() {
    // Drop zone click to trigger file input
    dropZone.addEventListener('click', () => {
        fileInput.click();
    });

    // File input change
    fileInput.addEventListener('change', handleFileSelect);

    // Drag and drop events
    dropZone.addEventListener('dragover', handleDragOver);
    dropZone.addEventListener('dragleave', handleDragLeave);
    dropZone.addEventListener('drop', handleDrop);

    // Compare button
    compareBtn.addEventListener('click', handleCompare);
}

// Drag and drop handlers
function handleDragOver(e) {
    e.preventDefault();
    e.stopPropagation();
    dropZone.classList.add('drag-over');
}

function handleDragLeave(e) {
    e.preventDefault();
    e.stopPropagation();
    dropZone.classList.remove('drag-over');
}

function handleDrop(e) {
    e.preventDefault();
    e.stopPropagation();
    dropZone.classList.remove('drag-over');

    const files = e.dataTransfer.files;
    if (files.length > 0) {
        handleFile(files[0]);
    }
}

// File selection handler
function handleFileSelect(e) {
    const files = e.target.files;
    if (files.length > 0) {
        handleFile(files[0]);
    }
}

// File validation and display
function handleFile(file) {
    // Validate file format
    const fileExtension = '.' + file.name.split('.').pop().toLowerCase();
    if (!SUPPORTED_FORMATS.includes(fileExtension)) {
        showToast(`Unsupported file format. Please use: ${SUPPORTED_FORMATS.join(', ')}`, 'error');
        return;
    }

    // Validate file size
    if (file.size > MAX_FILE_SIZE) {
        showToast(`File too large. Maximum size is ${MAX_FILE_SIZE / (1024 * 1024)}MB`, 'error');
        return;
    }

    // Display file info
    fileName.textContent = file.name;
    fileSize.textContent = formatFileSize(file.size);
    fileInfo.classList.remove('hidden');

    // Upload file
    uploadFile(file);
}

// Format file size for display
function formatFileSize(bytes) {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return Math.round(bytes / Math.pow(k, i) * 100) / 100 + ' ' + sizes[i];
}

// Show loading indicator
function showLoading() {
    loadingIndicator.classList.remove('hidden');
    dropZone.style.opacity = '0.5';
    dropZone.style.pointerEvents = 'none';
}

// Hide loading indicator
function hideLoading() {
    loadingIndicator.classList.add('hidden');
    dropZone.style.opacity = '1';
    dropZone.style.pointerEvents = 'auto';
}

// Initialize on page load
document.addEventListener('DOMContentLoaded', init);

// Upload file to API
async function uploadFile(file) {
    showLoading();

    const formData = new FormData();
    formData.append('audio_file', file);  // Changed from 'file' to 'audio_file'

    try {
        const startTime = Date.now();
        const response = await fetch('/api/generate', {
            method: 'POST',
            body: formData
        });

        const processingTime = ((Date.now() - startTime) / 1000).toFixed(2);

        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.error?.message || 'Failed to generate image');
        }

        const data = await response.json();
        
        // Add processing time from response or calculated time
        data.processing_time = data.processing_time || processingTime;
        data.filename = file.name;

        // Add to generated images
        state.generatedImages.push(data);

        // Display the generated image
        displayGeneratedImage(data);

        // Show success message
        showToast(`Image generated successfully in ${data.processing_time}s`, 'success');

        // Reset file input
        fileInput.value = '';
        fileInfo.classList.add('hidden');

    } catch (error) {
        console.error('Upload error:', error);
        showToast(error.message, 'error');
    } finally {
        hideLoading();
    }
}

// Display generated image in results grid
function displayGeneratedImage(imageData) {
    // Remove empty state if present
    const emptyState = resultsGrid.querySelector('.empty-state');
    if (emptyState) {
        emptyState.remove();
    }

    // Create image card
    const card = document.createElement('div');
    card.className = 'image-card';
    card.dataset.imageId = imageData.image_id;

    const transcriptionHtml = imageData.transcribed_text 
        ? `<div class="image-card-transcription">"${imageData.transcribed_text}"</div>`
        : '';

    card.innerHTML = `
        <div class="image-card-header">
            <input type="checkbox" class="image-card-checkbox" data-image-id="${imageData.image_id}">
            <img src="${imageData.image_url}" alt="Generated image">
        </div>
        <div class="image-card-body">
            <div class="image-card-title">${imageData.filename || 'Generated Image'}</div>
            <div class="image-card-meta">
                <span>ID: ${imageData.image_id.substring(0, 8)}...</span>
                <span>${imageData.processing_time}s</span>
            </div>
            ${transcriptionHtml}
        </div>
    `;

    // Add checkbox event listener
    const checkbox = card.querySelector('.image-card-checkbox');
    checkbox.addEventListener('change', handleCheckboxChange);

    // Add to grid
    resultsGrid.appendChild(card);
}

// Handle checkbox selection
function handleCheckboxChange(e) {
    const imageId = e.target.dataset.imageId;
    
    if (e.target.checked) {
        if (state.selectedImages.size >= 2) {
            // Uncheck the oldest selection
            const firstSelected = Array.from(state.selectedImages)[0];
            state.selectedImages.delete(firstSelected);
            const firstCheckbox = document.querySelector(`input[data-image-id="${firstSelected}"]`);
            if (firstCheckbox) {
                firstCheckbox.checked = false;
            }
        }
        state.selectedImages.add(imageId);
    } else {
        state.selectedImages.delete(imageId);
    }

    updateCompareButton();
}

// Update compare button state
function updateCompareButton() {
    const count = state.selectedImages.size;
    compareBtn.textContent = `Compare Selected (${count}/2)`;
    compareBtn.disabled = count !== 2;
}

// Handle compare button click
async function handleCompare() {
    if (state.selectedImages.size !== 2) {
        showToast('Please select exactly 2 images to compare', 'warning');
        return;
    }

    const [imageId1, imageId2] = Array.from(state.selectedImages);

    try {
        compareBtn.disabled = true;
        compareBtn.textContent = 'Comparing...';

        const response = await fetch('/api/compare', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                image_id_1: imageId1,
                image_id_2: imageId2
            })
        });

        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.error?.message || 'Failed to compare images');
        }

        const data = await response.json();

        // Display comparison results
        displayComparison(imageId1, imageId2, data.similarity_score);

        // Show success message
        showToast('Comparison complete!', 'success');

    } catch (error) {
        console.error('Comparison error:', error);
        showToast(error.message, 'error');
    } finally {
        compareBtn.disabled = false;
        updateCompareButton();
    }
}

// Display comparison results
function displayComparison(imageId1, imageId2, similarityScore) {
    // Get image data
    const image1 = state.generatedImages.find(img => img.image_id === imageId1);
    const image2 = state.generatedImages.find(img => img.image_id === imageId2);

    if (!image1 || !image2) {
        showToast('Could not find image data', 'error');
        return;
    }

    // Update comparison section
    document.getElementById('compare-img-1').src = image1.image_url;
    document.getElementById('compare-img-2').src = image2.image_url;
    document.getElementById('compare-label-1').textContent = image1.filename || 'Image 1';
    document.getElementById('compare-label-2').textContent = image2.filename || 'Image 2';

    // Update similarity score
    const percentage = Math.round(similarityScore * 100);
    const scoreElement = document.getElementById('similarity-score');
    const barElement = document.getElementById('similarity-bar');

    scoreElement.textContent = `${percentage}%`;
    barElement.style.width = `${percentage}%`;

    // Apply color coding
    scoreElement.className = 'similarity-score';
    if (percentage < 30) {
        scoreElement.classList.add('similarity-low');
    } else if (percentage < 70) {
        scoreElement.classList.add('similarity-medium');
    } else {
        scoreElement.classList.add('similarity-high');
    }

    // Show comparison section
    comparisonSection.classList.remove('hidden');

    // Scroll to comparison section
    comparisonSection.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

// Toast notification system
function showToast(message, type = 'info') {
    const toastContainer = document.getElementById('toast-container');
    
    const toast = document.createElement('div');
    toast.className = `toast ${type}`;
    
    toast.innerHTML = `
        <div class="toast-message">${message}</div>
        <button class="toast-close" aria-label="Close">&times;</button>
    `;
    
    // Add close button handler
    const closeBtn = toast.querySelector('.toast-close');
    closeBtn.addEventListener('click', () => {
        removeToast(toast);
    });
    
    toastContainer.appendChild(toast);
    
    // Auto-remove after 5 seconds
    setTimeout(() => {
        removeToast(toast);
    }, 5000);
}

// Remove toast with animation
function removeToast(toast) {
    toast.style.animation = 'slideIn 0.3s ease reverse';
    setTimeout(() => {
        if (toast.parentNode) {
            toast.parentNode.removeChild(toast);
        }
    }, 300);
}

// Enhanced error handling for fetch requests
async function fetchWithErrorHandling(url, options = {}) {
    try {
        const response = await fetch(url, options);
        
        if (!response.ok) {
            // Try to parse error response
            let errorMessage = 'An error occurred';
            try {
                const errorData = await response.json();
                errorMessage = errorData.error?.message || errorData.message || errorMessage;
            } catch (e) {
                // If JSON parsing fails, use status text
                errorMessage = response.statusText || errorMessage;
            }
            
            // Distinguish between client and server errors
            if (response.status >= 400 && response.status < 500) {
                throw new Error(`Client Error: ${errorMessage}`);
            } else if (response.status >= 500) {
                throw new Error(`Server Error: ${errorMessage}`);
            } else {
                throw new Error(errorMessage);
            }
        }
        
        return response;
    } catch (error) {
        // Network errors
        if (error.name === 'TypeError' && error.message.includes('fetch')) {
            throw new Error('Network error: Unable to connect to server. Please check your connection.');
        }
        throw error;
    }
}

// Retry mechanism for failed uploads
async function uploadFileWithRetry(file, maxRetries = 2) {
    let lastError;
    
    for (let attempt = 0; attempt <= maxRetries; attempt++) {
        try {
            if (attempt > 0) {
                showToast(`Retrying upload (attempt ${attempt + 1}/${maxRetries + 1})...`, 'warning');
            }
            
            await uploadFile(file);
            return; // Success
        } catch (error) {
            lastError = error;
            
            if (attempt < maxRetries) {
                // Wait before retrying (exponential backoff)
                await new Promise(resolve => setTimeout(resolve, 1000 * Math.pow(2, attempt)));
            }
        }
    }
    
    // All retries failed
    showToast(`Upload failed after ${maxRetries + 1} attempts: ${lastError.message}`, 'error');
    throw lastError;
}

// Global error handler
window.addEventListener('error', (event) => {
    console.error('Global error:', event.error);
    showToast('An unexpected error occurred. Please refresh the page.', 'error');
});

// Handle unhandled promise rejections
window.addEventListener('unhandledrejection', (event) => {
    console.error('Unhandled promise rejection:', event.reason);
    showToast('An unexpected error occurred. Please try again.', 'error');
});
