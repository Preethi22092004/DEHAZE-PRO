document.addEventListener('DOMContentLoaded', function() {
    // Elements
    const uploadBtn = document.getElementById('upload-btn');
    const fileInput = document.getElementById('file-input');
    const processingSection = document.getElementById('processing-section');
    const resultsSection = document.getElementById('results-section');
    const progressBar = document.getElementById('progress-bar');
    const statusMessage = document.getElementById('status-message');
    const originalImage = document.getElementById('original-image');
    const processedImage = document.getElementById('processed-image');
    const downloadBtn = document.getElementById('download-btn');
    const tryAnotherBtn = document.getElementById('try-another-btn');
    const errorToast = document.getElementById('error-toast');
    const errorMessage = document.getElementById('error-message');
    
    // Initialize Bootstrap toast
    const toast = new bootstrap.Toast(errorToast);
    
    // Event listeners
    uploadBtn.addEventListener('click', function() {
        fileInput.click();
    });
    
    fileInput.addEventListener('change', function(e) {
        if (e.target.files.length > 0) {
            handleFileUpload(e.target.files[0]);
        }
    });
    
    tryAnotherBtn.addEventListener('click', function() {
        resetUI();
        uploadBtn.click();
    });
    
    // Handle file uploads via drag and drop
    document.addEventListener('dragover', function(e) {
        e.preventDefault();
        e.stopPropagation();
    });
    
    document.addEventListener('drop', function(e) {
        e.preventDefault();
        e.stopPropagation();
        
        if (e.dataTransfer.files.length > 0) {
            handleFileUpload(e.dataTransfer.files[0]);
        }
    });
    
    // Functions
    function handleFileUpload(file) {
        // Check if file is an image
        if (!file.type.match('image/jpeg') && !file.type.match('image/png') && !file.type.match('image/webp')) {
            showError('Please upload a JPEG, PNG, or WebP image.');
            return;
        }
        
        // Check file size (max 10MB)
        if (file.size > 10 * 1024 * 1024) {
            showError('File is too large. Maximum size is 10MB.');
            return;
        }
        
        // Show processing UI
        processingSection.classList.remove('d-none');
        resultsSection.classList.add('d-none');
        
        // Simulate progress
        let progress = 0;
        const progressInterval = setInterval(() => {
            progress += 5;
            if (progress >= 90) {
                clearInterval(progressInterval);
            }
            progressBar.style.width = `${progress}%`;
        }, 200);
        
        // Get selected model
        const selectedModel = document.querySelector('input[name="model"]:checked');
        const modelType = selectedModel ? selectedModel.value : 'enhanced';
        
        // Create FormData
        const formData = new FormData();
        formData.append('file', file);
        formData.append('model', modelType);
        
        // Display original image preview
        const reader = new FileReader();
        reader.onload = function(e) {
            originalImage.src = e.target.result;
        };
        reader.readAsDataURL(file);
        
        // Update status
        statusMessage.textContent = `Processing image with ${getModelName(modelType)} model...`;
        
        // Send request to server
        fetch('/upload-image', {
            method: 'POST',
            body: formData
        })
        .then(response => {
            if (!response.ok) {
                return response.json().then(data => {
                    throw new Error(data.error || 'Server error occurred');
                });
            }
            return response.json();
        })
        .then(data => {
            // Clear progress interval
            clearInterval(progressInterval);
            progressBar.style.width = '100%';
            
            // Show results
            setTimeout(() => {
                processingSection.classList.add('d-none');
                resultsSection.classList.remove('d-none');
                
                // Set processed image with cache busting
                const cacheBuster = new Date().getTime();
                processedImage.src = `/${data.output}?v=${cacheBuster}`;

                // Set download link
                downloadBtn.href = `/${data.output}`;
                downloadBtn.download = data.output.split('/').pop();
            }, 500);
        })
        .catch(error => {
            clearInterval(progressInterval);
            processingSection.classList.add('d-none');
            showError(error.message);
        });
    }
    
    function resetUI() {
        // Reset file input
        fileInput.value = '';
        
        // Reset progress
        progressBar.style.width = '0%';
        statusMessage.textContent = 'Uploading image...';
        
        // Hide sections
        processingSection.classList.add('d-none');
        resultsSection.classList.add('d-none');
    }
    
    function showError(message) {
        errorMessage.textContent = message;
        toast.show();
    }
    
    function getModelName(modelType) {
        switch(modelType) {
            case 'enhanced':
                return 'Enhanced Dehazing';
            case 'aod':
                return 'AOD-Net';
            case 'clahe':
                return 'CLAHE';
            default:
                return 'dehazing';
        }
    }
});
