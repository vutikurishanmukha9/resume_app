// DOM Elements
const uploadForm = document.getElementById('uploadForm');
const fileInput = document.getElementById('resumeFile');
const fileNameDisplay = document.getElementById('fileName');
const analyzeBtn = document.getElementById('analyzeBtn');
const btnText = analyzeBtn.querySelector('.btn-text');
const btnLoader = analyzeBtn.querySelector('.btn-loader');
const resultDiv = document.getElementById('result');
const errorDiv = document.getElementById('error');

// Constants
const MAX_FILE_SIZE = 16 * 1024 * 1024; // 16MB
const ALLOWED_TYPES = ['application/pdf', 'text/plain'];

/**
 * Display file name when selected
 */
fileInput.addEventListener('change', function(e) {
    const file = e.target.files[0];
    
    if (file) {
        // Validate file
        const validation = validateFile(file);
        
        if (!validation.valid) {
            showError(validation.error);
            fileInput.value = '';
            fileNameDisplay.classList.remove('active');
            return;
        }
        
        // Display file name
        fileNameDisplay.textContent = `📎 ${file.name} (${formatFileSize(file.size)})`;
        fileNameDisplay.classList.add('active');
        hideError();
    } else {
        fileNameDisplay.classList.remove('active');
    }
});

/**
 * Validate uploaded file
 */
function validateFile(file) {
    // Check file size
    if (file.size > MAX_FILE_SIZE) {
        return {
            valid: false,
            error: 'File size exceeds 16MB limit. Please upload a smaller file.'
        };
    }
    
    // Check file type
    if (!ALLOWED_TYPES.includes(file.type)) {
        return {
            valid: false,
            error: 'Invalid file type. Only PDF and TXT files are allowed.'
        };
    }
    
    return { valid: true };
}

/**
 * Format file size for display
 */
function formatFileSize(bytes) {
    if (bytes === 0) return '0 Bytes';
    
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    
    return Math.round(bytes / Math.pow(k, i) * 100) / 100 + ' ' + sizes[i];
}

/**
 * Handle form submission
 */
uploadForm.addEventListener('submit', async function(e) {
    e.preventDefault();
    
    // Clear previous results and errors
    hideResult();
    hideError();
    
    // Get selected file
    const file = fileInput.files[0];
    
    if (!file) {
        showError('Please select a file to upload.');
        return;
    }
    
    // Validate file before upload
    const validation = validateFile(file);
    if (!validation.valid) {
        showError(validation.error);
        return;
    }
    
    // Prepare form data
    const formData = new FormData();
    formData.append('resume', file);
    
    // Show loading state
    setLoadingState(true);
    
    try {
        // Upload and analyze resume
        const response = await fetch('/upload', {
            method: 'POST',
            body: formData
        });
        
        const data = await response.json();
        
        if (!response.ok) {
            throw new Error(data.error || 'Failed to analyze resume');
        }
        
        // Display results
        displayResults(data);
        
    } catch (error) {
        console.error('Error:', error);
        showError(error.message || 'An error occurred while processing your resume. Please try again.');
    } finally {
        setLoadingState(false);
    }
});

/**
 * Set loading state for button
 */
function setLoadingState(isLoading) {
    if (isLoading) {
        analyzeBtn.disabled = true;
        btnText.style.display = 'none';
        btnLoader.style.display = 'inline-flex';
    } else {
        analyzeBtn.disabled = false;
        btnText.style.display = 'inline';
        btnLoader.style.display = 'none';
    }
}

/**
 * Display analysis results
 */
function displayResults(data) {
    let html = `
        <h2>
            <span>🎯</span>
            Predicted Job Category
        </h2>
        <div class="job-badge">${escapeHtml(data.predicted_job)}</div>
        
        <h3>💼 Top Job Matches</h3>
        <ul class="match-list">
    `;
    
    // Add each job match
    data.matches.forEach((match, index) => {
        const percentage = (parseFloat(match.score) * 100).toFixed(1);
        
        html += `
            <li class="match-item">
                <div class="match-title">${index + 1}. ${escapeHtml(match.title)}</div>
                <div class="match-score">Similarity Score: ${match.score} (${percentage}%)</div>
                <div class="score-bar">
                    <div class="score-fill" style="width: ${percentage}%"></div>
                </div>
            </li>
        `;
    });
    
    html += `
        </ul>
        
        <h3>💰 Estimated Salary</h3>
        <div class="salary-display">
            <span>📊</span>
            <span class="salary-amount">${escapeHtml(data.salary)}</span>
        </div>
    `;
    
    resultDiv.innerHTML = html;
    resultDiv.style.display = 'block';
    
    // Smooth scroll to results
    resultDiv.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

/**
 * Show error message
 */
function showError(message) {
    errorDiv.innerHTML = `
        <strong>⚠️ Error:</strong> ${escapeHtml(message)}
    `;
    errorDiv.style.display = 'block';
    
    // Scroll to error
    errorDiv.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

/**
 * Hide error message
 */
function hideError() {
    errorDiv.style.display = 'none';
    errorDiv.innerHTML = '';
}

/**
 * Hide result section
 */
function hideResult() {
    resultDiv.style.display = 'none';
    resultDiv.innerHTML = '';
}

/**
 * Escape HTML to prevent XSS
 */
function escapeHtml(text) {
    const map = {
        '&': '&amp;',
        '<': '&lt;',
        '>': '&gt;',
        '"': '&quot;',
        "'": '&#039;'
    };
    return text.toString().replace(/[&<>"']/g, m => map[m]);
}

/**
 * Reset form
 */
function resetForm() {
    uploadForm.reset();
    fileNameDisplay.classList.remove('active');
    hideResult();
    hideError();
}

// Optional: Add drag and drop functionality
const fileLabel = document.querySelector('.file-label');

fileLabel.addEventListener('dragover', function(e) {
    e.preventDefault();
    this.style.borderColor = '#667eea';
    this.style.background = '#edf2f7';
});

fileLabel.addEventListener('dragleave', function(e) {
    e.preventDefault();
    this.style.borderColor = '#cbd5e0';
    this.style.background = '#f7fafc';
});

fileLabel.addEventListener('drop', function(e) {
    e.preventDefault();
    this.style.borderColor = '#cbd5e0';
    this.style.background = '#f7fafc';
    
    const files = e.dataTransfer.files;
    if (files.length > 0) {
        fileInput.files = files;
        // Trigger change event
        const event = new Event('change', { bubbles: true });
        fileInput.dispatchEvent(event);
    }
});