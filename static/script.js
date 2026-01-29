document.addEventListener('DOMContentLoaded', () => {
    const uploadArea = document.getElementById('upload-area');
    const fileInput = document.getElementById('file-input');
    const previewContainer = document.getElementById('preview-container');
    const imagePreview = document.getElementById('image-preview');
    const removeBtn = document.getElementById('remove-btn');
    const analyzeBtn = document.getElementById('analyze-btn');
    const resultContainer = document.getElementById('result-container');
    const btnText = analyzeBtn.querySelector('.btn-text');
    const loader = analyzeBtn.querySelector('.loader');

    let currentFile = null;

    // Drag & Drop
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        uploadArea.addEventListener(eventName, preventDefaults, false);
    });

    function preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
    }

    ['dragenter', 'dragover'].forEach(eventName => {
        uploadArea.addEventListener(eventName, highlight, false);
    });

    ['dragleave', 'drop'].forEach(eventName => {
        uploadArea.addEventListener(eventName, unhighlight, false);
    });

    function highlight() {
        uploadArea.classList.add('dragover');
    }

    function unhighlight() {
        uploadArea.classList.remove('dragover');
    }

    uploadArea.addEventListener('drop', handleDrop, false);

    function handleDrop(e) {
        const dt = e.dataTransfer;
        const files = dt.files;
        handleFiles(files);
    }

    // Click to upload
    uploadArea.addEventListener('click', (e) => {
        if (e.target !== removeBtn && e.target.parentElement !== removeBtn) {
            fileInput.click();
        }
    });

    fileInput.addEventListener('change', function() {
        handleFiles(this.files);
    });

    function handleFiles(files) {
        if (files.length > 0) {
            const file = files[0];
            if (file.type.startsWith('image/')) {
                currentFile = file;
                showPreview(file);
                analyzeBtn.disabled = false;
                resultContainer.style.display = 'none';
            } else {
                alert('Please upload an image file.');
            }
        }
    }

    function showPreview(file) {
        const reader = new FileReader();
        reader.readAsDataURL(file);
        reader.onloadend = function() {
            imagePreview.src = reader.result;
            previewContainer.style.display = 'flex';
            uploadArea.querySelector('.upload-content').style.visibility = 'hidden';
        }
    }

    removeBtn.addEventListener('click', (e) => {
        e.stopPropagation(); // Prevent triggering upload click
        currentFile = null;
        fileInput.value = '';
        previewContainer.style.display = 'none';
        uploadArea.querySelector('.upload-content').style.visibility = 'visible';
        analyzeBtn.disabled = true;
        resultContainer.style.display = 'none';
    });

    // Analyze
    analyzeBtn.addEventListener('click', async () => {
        if (!currentFile) return;

        // UI Loading State
        analyzeBtn.disabled = true;
        btnText.style.display = 'none';
        loader.style.display = 'inline-block';
        resultContainer.style.display = 'none';

        const formData = new FormData();
        formData.append('file', currentFile);

        try {
            const response = await fetch('/predict', {
                method: 'POST',
                body: formData
            });

            const data = await response.json();

            if (response.ok) {
                displayResult(data);
            } else {
                alert('Error: ' + (data.error || 'Something went wrong'));
            }
        } catch (error) {
            console.error('Error:', error);
            alert('Failed to connect to the server.');
        } finally {
            // Reset Button State
            analyzeBtn.disabled = false;
            btnText.style.display = 'inline';
            loader.style.display = 'none';
        }
    });

    function displayResult(data) {
        const breedName = document.getElementById('breed-name');
        const confidenceValue = document.getElementById('confidence-value');
        const progressFill = document.getElementById('progress-fill');
        const demoBadge = document.getElementById('demo-badge');
        const demoMessage = document.getElementById('demo-message');

        breedName.textContent = data.breed;
        const confidencePercent = Math.round(data.confidence * 100);
        confidenceValue.textContent = `${confidencePercent}%`;
        
        // Reset width first to allow animation
        progressFill.style.width = '0%';
        resultContainer.style.display = 'block';

        // Trigger animation
        setTimeout(() => {
            progressFill.style.width = `${confidencePercent}%`;
        }, 100);

        if (data.is_demo) {
            demoBadge.style.display = 'inline-block';
            demoMessage.style.display = 'block';
            demoMessage.textContent = data.message;
        } else {
            demoBadge.style.display = 'none';
            demoMessage.style.display = 'none';
        }
    }
});
