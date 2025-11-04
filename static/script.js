// ==================== DOM ELEMENTS ====================
const uploadForm = document.getElementById('uploadForm');
const fileInput = document.getElementById('resumeFile');
const fileNameDisplay = document.getElementById('fileName');
const analyzeBtn = document.getElementById('analyzeBtn');
const matchBtn = document.getElementById('matchBtn');
const jdTextArea = document.getElementById('jdText');
const resultDiv = document.getElementById('result');
const jdMatchResult = document.getElementById('jdMatchResult');
const errorDiv = document.getElementById('error');

// ==================== CONSTANTS ====================
const MAX_FILE_SIZE = 16 * 1024 * 1024; // 16MB
const ALLOWED_TYPES = ['application/pdf', 'text/plain'];
const MIN_JD_LENGTH = 20;
const RETRY_DELAY = 2000; // 2 seconds

// ==================== FILE VALIDATION ====================

/**
 * Validate uploaded file
 * @param {File} file - The file to validate
 * @returns {Object} Validation result with valid flag and error message
 */
function validateFile(file) {
    if (!file) {
        return {
            valid: false,
            error: 'No file selected. Please choose a file to upload.'
        };
    }

    // Check file size
    if (file.size === 0) {
        return {
            valid: false,
            error: 'The selected file is empty. Please choose a valid file.'
        };
    }

    if (file.size > MAX_FILE_SIZE) {
        return {
            valid: false,
            error: `File size (${formatFileSize(file.size)}) exceeds the 16MB limit. Please upload a smaller file.`
        };
    }

    // Check file type
    if (!ALLOWED_TYPES.includes(file.type)) {
        return {
            valid: false,
            error: 'Invalid file type. Only PDF and TXT files are supported.'
        };
    }

    return { valid: true };
}

/**
 * Format file size for display
 * @param {number} bytes - File size in bytes
 * @returns {string} Formatted file size
 */
function formatFileSize(bytes) {
    if (bytes === 0) return '0 Bytes';

    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));

    return Math.round((bytes / Math.pow(k, i)) * 100) / 100 + ' ' + sizes[i];
}

// ==================== EVENT LISTENERS ====================

/**
 * Handle file input change
 */
fileInput.addEventListener('change', function(e) {
    const file = e.target.files[0];

    if (file) {
        // Validate file
        const validation = validateFile(file);

        if (!validation.valid) {
            showError(validation.error);
            fileInput.value = '';
            fileNameDisplay.textContent = '';
            fileNameDisplay.classList.remove('active');
            return;
        }

        // Display file information
        fileNameDisplay.textContent = `📎 ${file.name} (${formatFileSize(file.size)})`;
        fileNameDisplay.classList.add('active');
        hideError();
    } else {
        fileNameDisplay.textContent = '';
        fileNameDisplay.classList.remove('active');
    }
});

/**
 * Handle form submission for resume analysis
 */
uploadForm.addEventListener('submit', async function(e) {
    e.preventDefault();

    // Get selected file
    const file = fileInput.files[0];

    if (!file) {
        showError('Please select a resume file to upload.');
        return;
    }

    // Validate file
    const validation = validateFile(file);
    if (!validation.valid) {
        showError(validation.error);
        return;
    }

    // Clear previous results
    resetResults();

    // Prepare form data
    const formData = new FormData();
    formData.append('resume', file);

    // Show loading state
    setButtonLoading(analyzeBtn, true);

    try {
        // Upload and analyze resume
        const response = await fetch('/upload', {
            method: 'POST',
            body: formData
        });

        const data = await response.json();

        if (!response.ok) {
            throw new Error(data.error || `Server error (${response.status})`);
        }

        if (data.success) {
            displayResults(data);
        } else {
            showError(data.error || 'Failed to analyze resume. Please try again.');
        }

    } catch (error) {
        console.error('Analysis error:', error);
        handleFetchError(error, 'analyze');
    } finally {
        setButtonLoading(analyzeBtn, false);
    }
});

/**
 * Handle JD match button click
 */
matchBtn.addEventListener('click', async function() {
    const file = fileInput.files[0];
    const jdText = jdTextArea.value.trim();

    // Validate inputs
    if (!file) {
        showError('Please select a resume file first.');
        fileInput.focus();
        return;
    }

    if (!jdText) {
        showError('Please paste a Job Description to check the match.');
        jdTextArea.focus();
        return;
    }

    if (jdText.length < MIN_JD_LENGTH) {
        showError(`Job Description is too short. Please provide at least ${MIN_JD_LENGTH} characters.`);
        jdTextArea.focus();
        return;
    }

    // Validate file
    const validation = validateFile(file);
    if (!validation.valid) {
        showError(validation.error);
        return;
    }

    // Clear previous results
    resetResults();

    // Prepare form data
    const formData = new FormData();
    formData.append('resume', file);
    formData.append('jd_text', jdText);

    // Show loading state
    setButtonLoading(matchBtn, true);

    try {
        // Calculate JD match
        const response = await fetch('/match_jd_resume', {
            method: 'POST',
            body: formData
        });

        const data = await response.json();

        if (!response.ok) {
            throw new Error(data.error || `Server error (${response.status})`);
        }

        if (data.success) {
            displayJDMatch(data);
        } else {
            showError(data.error || 'Failed to calculate JD match. Please try again.');
        }

    } catch (error) {
        console.error('JD match error:', error);
        handleFetchError(error, 'match');
    } finally {
        setButtonLoading(matchBtn, false);
    }
});

// ==================== UI CONTROL FUNCTIONS ====================

/**
 * Set loading state for a button
 * @param {HTMLElement} button - Button element
 * @param {boolean} isLoading - Loading state
 */
function setButtonLoading(button, isLoading) {
    const btnText = button.querySelector('.btn-text');
    const btnLoader = button.querySelector('.btn-loader');

    if (isLoading) {
        button.disabled = true;
        btnText.style.display = 'none';
        btnLoader.style.display = 'inline-flex';
    } else {
        button.disabled = false;
        btnText.style.display = 'inline';
        btnLoader.style.display = 'none';
    }
}

/**
 * Reset all result sections
 */
function resetResults() {
    resultDiv.style.display = 'none';
    resultDiv.innerHTML = '';
    jdMatchResult.style.display = 'none';
    jdMatchResult.innerHTML = '';
    hideError();
}

/**
 * Show error message
 * @param {string} message - Error message to display
 */
function showError(message) {
    errorDiv.innerHTML = `<strong>⚠️ Error:</strong> ${escapeHtml(message)}`;
    errorDiv.style.display = 'block';
    
    // Smooth scroll to error
    errorDiv.scrollIntoView({ 
        behavior: 'smooth', 
        block: 'nearest' 
    });
}

/**
 * Hide error message
 */
function hideError() {
    errorDiv.style.display = 'none';
    errorDiv.innerHTML = '';
}

/**
 * Handle fetch errors with user-friendly messages
 * @param {Error} error - The error object
 * @param {string} action - The action being performed
 */
function handleFetchError(error, action) {
    let message = '';

    if (error.name === 'TypeError' && error.message.includes('fetch')) {
        message = 'Unable to connect to the server. Please check your internet connection and try again.';
    } else if (error.message.includes('503')) {
        message = 'The system is still initializing. Please wait a moment and try again.';
    } else if (error.message.includes('413')) {
        message = 'File size is too large. Please upload a file smaller than 16MB.';
    } else if (error.message.includes('500')) {
        message = 'A server error occurred. Please try again later.';
    } else {
        message = error.message || `An error occurred while ${action === 'analyze' ? 'analyzing the resume' : 'calculating the match'}. Please try again.`;
    }

    showError(message);
}

// ==================== DISPLAY FUNCTIONS ====================

/**
 * Display analysis results
 * @param {Object} data - Response data from server
 */
function displayResults(data) {
    if (!data || !data.predicted_job || !data.matches || !data.salary) {
        showError('Invalid response from server. Please try again.');
        return;
    }

    let html = `
        <h2>📋 Analysis Results</h2>
        
        <div class="result-card">
            <h3>🎯 Predicted Job Category</h3>
            <div class="job-badge">${escapeHtml(data.predicted_job)}</div>
        </div>
        
        <div class="result-card">
            <h3>💼 Top Job Matches</h3>
            <ul class="match-list">
    `;

    // Add each job match
    if (Array.isArray(data.matches) && data.matches.length > 0) {
        data.matches.forEach((match, index) => {
            const score = parseFloat(match.score) || 0;
            const percentage = (score * 100).toFixed(1);

            html += `
                <li class="match-item">
                    <div class="match-header">
                        <span class="match-rank">#${index + 1}</span>
                        <span class="match-title">${escapeHtml(match.title)}</span>
                    </div>
                    <div class="match-score">Similarity: ${match.score} (${percentage}%)</div>
                    <div class="score-bar">
                        <div class="score-fill" style="width: ${percentage}%"></div>
                    </div>
                </li>
            `;
        });
    } else {
        html += '<li class="match-item">No matches found.</li>';
    }

    html += `
            </ul>
        </div>
        
        <div class="result-card">
            <h3>💰 Estimated Salary</h3>
            <div class="salary-display">
                <span class="salary-icon">💵</span>
                <span class="salary-amount">${escapeHtml(data.salary)}</span>
            </div>
        </div>
    `;

    resultDiv.innerHTML = html;
    resultDiv.style.display = 'block';

    // Smooth scroll to results
    setTimeout(() => {
        resultDiv.scrollIntoView({ 
            behavior: 'smooth', 
            block: 'nearest' 
        });
    }, 100);
}

/**
 * Display JD match results
 * @param {Object} data - Response data from server
 */
function displayJDMatch(data) {
    if (!data || typeof data.match_percentage !== 'number') {
        showError('Invalid response from server. Please try again.');
        return;
    }

    const percentage = data.match_percentage;
    let matchLevel = '';
    let matchClass = '';

    // Determine match level
    if (percentage >= 80) {
        matchLevel = 'Excellent Match! 🎉';
        matchClass = 'excellent';
    } else if (percentage >= 60) {
        matchLevel = 'Good Match! 👍';
        matchClass = 'good';
    } else if (percentage >= 40) {
        matchLevel = 'Moderate Match 👌';
        matchClass = 'moderate';
    } else {
        matchLevel = 'Low Match 🤔';
        matchClass = 'low';
    }

    const html = `
        <h2>📊 JD Match Analysis</h2>
        <div class="match-card ${matchClass}">
            <div class="match-percentage-large">${percentage}%</div>
            <div class="match-level">${matchLevel}</div>
            <div class="match-description">${escapeHtml(data.message || '')}</div>
            <div class="match-bar-container">
                <div class="match-bar">
                    <div class="match-bar-fill" style="width: ${percentage}%"></div>
                </div>
            </div>
        </div>
    `;

    jdMatchResult.innerHTML = html;
    jdMatchResult.style.display = 'block';

    // Smooth scroll to results
    setTimeout(() => {
        jdMatchResult.scrollIntoView({ 
            behavior: 'smooth', 
            block: 'nearest' 
        });
    }, 100);
}

// ==================== UTILITY FUNCTIONS ====================

/**
 * Escape HTML to prevent XSS attacks
 * @param {string} text - Text to escape
 * @returns {string} Escaped text
 */
function escapeHtml(text) {
    const map = {
        '&': '&amp;',
        '<': '&lt;',
        '>': '&gt;',
        '"': '&quot;',
        "'": '&#039;'
    };
    return String(text).replace(/[&<>"']/g, m => map[m]);
}

/**
 * Reset the entire form
 */
function resetForm() {
    uploadForm.reset();
    fileNameDisplay.textContent = '';
    fileNameDisplay.classList.remove('active');
    resetResults();
}

// ==================== DRAG & DROP FUNCTIONALITY ====================

const fileLabel = document.querySelector('.file-label');

if (fileLabel) {
    // Drag over effect
    ['dragover', 'dragenter'].forEach(eventName => {
        fileLabel.addEventListener(eventName, function(e) {
            e.preventDefault();
            e.stopPropagation();
            this.classList.add('drag-over');
        });
    });

    // Drag leave effect
    ['dragleave', 'dragend'].forEach(eventName => {
        fileLabel.addEventListener(eventName, function(e) {
            e.preventDefault();
            e.stopPropagation();
            this.classList.remove('drag-over');
        });
    });

    // Drop handler
    fileLabel.addEventListener('drop', function(e) {
        e.preventDefault();
        e.stopPropagation();
        this.classList.remove('drag-over');

        const files = e.dataTransfer.files;
        
        if (files.length > 0) {
            // Set the file input
            fileInput.files = files;
            
            // Trigger change event
            const event = new Event('change', { bubbles: true });
            fileInput.dispatchEvent(event);
        }
    });
}

// ==================== KEYBOARD SHORTCUTS ====================

document.addEventListener('keydown', function(e) {
    // Ctrl/Cmd + Enter to analyze
    if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
        if (!analyzeBtn.disabled && fileInput.files.length > 0) {
            e.preventDefault();
            uploadForm.dispatchEvent(new Event('submit'));
        }
    }
    
    // Escape to reset form
    if (e.key === 'Escape') {
        resetForm();
    }
});

// ==================== INITIALIZATION ====================

// Check if browser supports required APIs
if (!window.FormData || !window.fetch) {
    showError('Your browser does not support required features. Please use a modern browser.');
}

// Log initialization
console.log('AI Resume Analyzer initialized successfully');