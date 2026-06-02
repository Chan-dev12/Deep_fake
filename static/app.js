document.addEventListener('DOMContentLoaded', () => {
    const dropZone = document.getElementById('dropZone');
    const fileInput = document.getElementById('fileInput');
    const browseBtn = document.getElementById('browseBtn');
    
    const uploadSection = document.getElementById('uploadSection');
    const scanningSection = document.getElementById('scanningSection');
    const resultsSection = document.getElementById('resultsSection');
    
    const scanPreview = document.getElementById('scanPreview');
    const croppedFace = document.getElementById('croppedFace');
    const mainCard = document.querySelector('.main-card');
    
    const resultVerdict = document.getElementById('resultVerdict');
    const resultIcon = document.getElementById('resultIcon');
    const confidenceVal = document.getElementById('confidenceVal');
    const confidenceBar = document.getElementById('confidenceBar');
    const rawScore = document.getElementById('rawScore');
    const resetBtn = document.getElementById('resetBtn');

    // Heatmap elements
    const heatmapFrame = document.getElementById('heatmapFrame');
    const heatmapImage = document.getElementById('heatmapImage');

    // Trigger click on input when clicking browse button
    browseBtn.addEventListener('click', () => {
        fileInput.click();
    });

    // Handle Drag & Drop highlights
    ['dragenter', 'dragover'].forEach(eventName => {
        dropZone.addEventListener(eventName, (e) => {
            e.preventDefault();
            dropZone.classList.add('drag-over');
        }, false);
    });

    ['dragleave', 'drop'].forEach(eventName => {
        dropZone.addEventListener(eventName, (e) => {
            e.preventDefault();
            dropZone.classList.remove('drag-over');
        }, false);
    });

    // Handle File Drop
    dropZone.addEventListener('drop', (e) => {
        const dt = e.dataTransfer;
        const files = dt.files;
        if (files.length > 0) {
            handleFileUpload(files[0]);
        }
    });

    // Handle File Selector Select
    fileInput.addEventListener('change', (e) => {
        if (e.target.files.length > 0) {
            handleFileUpload(e.target.files[0]);
        }
    });

    // Reset button functionality
    resetBtn.addEventListener('click', () => {
        // Clear inputs
        fileInput.value = '';
        
        // Remove state classes from card
        mainCard.classList.remove('is-real', 'is-fake');
        
        // Reset view states
        resultsSection.classList.add('hidden');
        scanningSection.classList.add('hidden');
        uploadSection.classList.remove('hidden');
        heatmapFrame.classList.add('hidden');
        
        // Reset image sources
        scanPreview.src = '';
        croppedFace.src = '';
        heatmapImage.src = '';
        confidenceBar.style.width = '0%';
    });

    function handleFileUpload(file) {
        if (!file.type.startsWith('image/')) {
            alert('Please select an image file (PNG, JPG, WEBP).');
            return;
        }

        // Show Scanning Preview
        const reader = new FileReader();
        reader.onload = (e) => {
            scanPreview.src = e.target.result;
            // Switch panels
            uploadSection.classList.add('hidden');
            scanningSection.classList.remove('hidden');
        };
        reader.readAsDataURL(file);

        // Prepare request
        const formData = new FormData();
        formData.append('file', file);

        // Simulate minimum scan duration for aesthetics (2 seconds)
        const scanStart = Date.now();

        fetch('/predict', {
            method: 'POST',
            body: formData
        })
        .then(response => response.json())
        .then(data => {
            const timeElapsed = Date.now() - scanStart;
            const remainingTime = Math.max(0, 2000 - timeElapsed);

            // Wait until minimum scan time has passed for smooth user experience
            setTimeout(() => {
                displayResults(data);
            }, remainingTime);
        })
        .catch(err => {
            setTimeout(() => {
                alert('Analysis failed. Make sure Flask server is running and the model is trained.');
                resetBtn.click();
            }, 1000);
        });
    }

    function displayResults(data) {
        // Hide scanning panel
        scanningSection.classList.add('hidden');

        if (data.error) {
            alert('Error: ' + data.error);
            resetBtn.click();
            return;
        }

        if (!data.detected) {
            alert(data.message || 'No face detected in the image.');
            resetBtn.click();
            return;
        }

        // Apply classification color theme
        if (data.label === 'REAL') {
            mainCard.classList.add('is-real');
            mainCard.classList.remove('is-fake');
            resultVerdict.textContent = 'REAL / PRISTINE';
            resultIcon.className = 'fa-solid fa-shield-halved';
        } else {
            mainCard.classList.add('is-fake');
            mainCard.classList.remove('is-real');
            resultVerdict.textContent = 'SYNTHETIC / FAKE';
            resultIcon.className = 'fa-solid fa-triangle-exclamation';
        }

        // Set metrics
        croppedFace.src = data.crop_url;
        confidenceVal.textContent = data.confidence + '%';
        rawScore.textContent = data.raw_score.toFixed(4);

        // Handle side-by-side heatmap displaying
        if (data.has_heatmap) {
            heatmapImage.src = data.heatmap_url;
            heatmapFrame.classList.remove('hidden');
        } else {
            heatmapFrame.classList.add('hidden');
        }

        // Show results panel
        resultsSection.classList.remove('hidden');

        // Animate the progress bar fill
        setTimeout(() => {
            confidenceBar.style.width = data.confidence + '%';
        }, 100);
    }
});
