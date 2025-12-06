/**
 * AI Resume Analyzer - Frontend Application
 * Optimized and Enhanced Version
 */

(function () {
    'use strict';

    // ==================== CONFIGURATION ====================
    const CONFIG = {
        MAX_FILE_SIZE: 16 * 1024 * 1024, // 16MB
        ALLOWED_TYPES: ['application/pdf', 'text/plain'],
        MIN_JD_LENGTH: 20,
        ENDPOINTS: {
            UPLOAD: '/upload',
            MATCH: '/match_jd_resume'
        },
        SCROLL_OPTIONS: {
            behavior: 'smooth',
            block: 'nearest'
        }
    };

    // ==================== DOM CACHE ====================
    const DOM = {
        uploadForm: null,
        fileInput: null,
        fileName: null,
        analyzeBtn: null,
        matchBtn: null,
        jdText: null,
        result: null,
        jdMatchResult: null,
        error: null,
        fileLabel: null
    };

    // ==================== INITIALIZATION ====================
    /**
     * Initialize application when DOM is ready
     */
    function init() {
        // Cache DOM elements
        cacheDOMElements();

        // Check browser compatibility
        if (!checkBrowserCompatibility()) {
            showError('Your browser does not support required features. Please use a modern browser.');
            return;
        }

        // Attach event listeners
        attachEventListeners();

        console.log('AI Resume Analyzer initialized successfully');
    }

    /**
     * Cache all DOM elements for performance
     */
    function cacheDOMElements() {
        DOM.uploadForm = document.getElementById('uploadForm');
        DOM.fileInput = document.getElementById('resumeFile');
        DOM.fileName = document.getElementById('fileName');
        DOM.analyzeBtn = document.getElementById('analyzeBtn');
        DOM.matchBtn = document.getElementById('matchBtn');
        DOM.jdText = document.getElementById('jdText');
        DOM.result = document.getElementById('result');
        DOM.jdMatchResult = document.getElementById('jdMatchResult');
        DOM.error = document.getElementById('error');
        DOM.fileLabel = document.querySelector('.file-label');
    }

    /**
     * Check if browser supports required APIs
     */
    function checkBrowserCompatibility() {
        return !!(window.FormData && window.fetch && window.File);
    }

    // ==================== EVENT LISTENERS ====================
    /**
     * Attach all event listeners
     */
    function attachEventListeners() {
        // File input change
        DOM.fileInput.addEventListener('change', handleFileChange);

        // Form submission for analysis
        DOM.uploadForm.addEventListener('submit', handleAnalyze);

        // JD match button
        DOM.matchBtn.addEventListener('click', handleJDMatch);

        // Drag and drop
        setupDragAndDrop();

        // Keyboard shortcuts
        setupKeyboardShortcuts();

        // File label keyboard accessibility
        DOM.fileLabel.addEventListener('keydown', handleFileLabelKeydown);
    }

    /**
     * Handle file input change
     */
    function handleFileChange(e) {
        const file = e.target.files[0];

        if (!file) {
            clearFileDisplay();
            return;
        }

        // Validate file
        const validation = validateFile(file);

        if (!validation.valid) {
            showError(validation.error);
            clearFileInput();
            return;
        }

        // Display file info
        displayFileInfo(file);
        hideError();
    }

    /**
     * Handle resume analysis
     */
    async function handleAnalyze(e) {
        e.preventDefault();

        const file = DOM.fileInput.files[0];

        if (!file) {
            showError('Please select a resume file to upload.');
            return;
        }

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
        setButtonLoading(DOM.analyzeBtn, true);

        try {
            const response = await fetch(CONFIG.ENDPOINTS.UPLOAD, {
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
            setButtonLoading(DOM.analyzeBtn, false);
        }
    }

    /**
     * Handle JD match
     */
    async function handleJDMatch() {
        const file = DOM.fileInput.files[0];
        const jdText = DOM.jdText.value.trim();

        // Validate inputs
        if (!file) {
            showError('Please select a resume file first.');
            DOM.fileInput.focus();
            return;
        }

        if (!jdText) {
            showError('Please paste a Job Description to check the match.');
            DOM.jdText.focus();
            return;
        }

        if (jdText.length < CONFIG.MIN_JD_LENGTH) {
            showError(`Job Description is too short. Please provide at least ${CONFIG.MIN_JD_LENGTH} characters.`);
            DOM.jdText.focus();
            return;
        }

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
        setButtonLoading(DOM.matchBtn, true);

        try {
            const response = await fetch(CONFIG.ENDPOINTS.MATCH, {
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
            setButtonLoading(DOM.matchBtn, false);
        }
    }

    /**
     * Handle file label keyboard interaction
     */
    function handleFileLabelKeydown(e) {
        if (e.key === 'Enter' || e.key === ' ') {
            e.preventDefault();
            DOM.fileInput.click();
        }
    }

    // ==================== DRAG AND DROP ====================
    /**
     * Setup drag and drop functionality
     */
    function setupDragAndDrop() {
        ['dragover', 'dragenter'].forEach(eventName => {
            DOM.fileLabel.addEventListener(eventName, handleDragOver);
        });

        ['dragleave', 'dragend'].forEach(eventName => {
            DOM.fileLabel.addEventListener(eventName, handleDragLeave);
        });

        DOM.fileLabel.addEventListener('drop', handleDrop);
    }

    function handleDragOver(e) {
        e.preventDefault();
        e.stopPropagation();
        DOM.fileLabel.classList.add('drag-over');
    }

    function handleDragLeave(e) {
        e.preventDefault();
        e.stopPropagation();
        DOM.fileLabel.classList.remove('drag-over');
    }

    function handleDrop(e) {
        e.preventDefault();
        e.stopPropagation();
        DOM.fileLabel.classList.remove('drag-over');

        const files = e.dataTransfer.files;

        if (files.length > 0) {
            DOM.fileInput.files = files;
            const event = new Event('change', { bubbles: true });
            DOM.fileInput.dispatchEvent(event);
        }
    }

    // ==================== KEYBOARD SHORTCUTS ====================
    /**
     * Setup keyboard shortcuts
     */
    function setupKeyboardShortcuts() {
        document.addEventListener('keydown', function (e) {
            // Ctrl/Cmd + Enter to analyze
            if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
                if (!DOM.analyzeBtn.disabled && DOM.fileInput.files.length > 0) {
                    e.preventDefault();
                    DOM.uploadForm.dispatchEvent(new Event('submit'));
                }
            }

            // Escape to reset
            if (e.key === 'Escape') {
                resetForm();
            }
        });
    }

    // ==================== VALIDATION ====================
    /**
     * Validate uploaded file
     */
    function validateFile(file) {
        if (!file) {
            return {
                valid: false,
                error: 'No file selected. Please choose a file to upload.'
            };
        }

        if (file.size === 0) {
            return {
                valid: false,
                error: 'The selected file is empty. Please choose a valid file.'
            };
        }

        if (file.size > CONFIG.MAX_FILE_SIZE) {
            return {
                valid: false,
                error: `File size (${formatFileSize(file.size)}) exceeds the 16MB limit. Please upload a smaller file.`
            };
        }

        if (!CONFIG.ALLOWED_TYPES.includes(file.type)) {
            return {
                valid: false,
                error: 'Invalid file type. Only PDF and TXT files are supported.'
            };
        }

        return { valid: true };
    }

    // ==================== UI CONTROL ====================
    /**
     * Set button loading state
     */
    function setButtonLoading(button, isLoading) {
        const btnText = button.querySelector('.btn-text');
        const btnLoader = button.querySelector('.btn-loader');

        button.disabled = isLoading;
        btnText.style.display = isLoading ? 'none' : 'inline';
        btnLoader.style.display = isLoading ? 'inline-flex' : 'none';
    }

    /**
     * Reset all results
     */
    function resetResults() {
        DOM.result.style.display = 'none';
        DOM.result.innerHTML = '';
        DOM.jdMatchResult.style.display = 'none';
        DOM.jdMatchResult.innerHTML = '';
        hideError();
    }

    /**
     * Reset entire form
     */
    function resetForm() {
        DOM.uploadForm.reset();
        clearFileDisplay();
        resetResults();
    }

    /**
     * Display file info
     */
    function displayFileInfo(file) {
        DOM.fileName.textContent = `${file.name} (${formatFileSize(file.size)})`;
        DOM.fileName.classList.add('active');
    }

    /**
     * Clear file display
     */
    function clearFileDisplay() {
        DOM.fileName.textContent = '';
        DOM.fileName.classList.remove('active');
    }

    /**
     * Clear file input
     */
    function clearFileInput() {
        DOM.fileInput.value = '';
        clearFileDisplay();
    }

    /**
     * Show error message
     */
    function showError(message) {
        DOM.error.innerHTML = `<strong>Error:</strong> ${escapeHtml(message)}`;
        DOM.error.style.display = 'block';
        scrollToElement(DOM.error);
    }

    /**
     * Hide error message
     */
    function hideError() {
        DOM.error.style.display = 'none';
        DOM.error.innerHTML = '';
    }

    /**
     * Handle fetch errors with user-friendly messages
     */
    function handleFetchError(error, action) {
        let message = '';

        if (error.name === 'TypeError' && error.message.includes('fetch')) {
            message = 'Unable to connect to the server. Please check your internet connection and try again.';
        } else if (error.message.includes('503')) {
            message = 'The system is still initializing. Please wait a moment and try again.';
        } else if (error.message.includes('413')) {
            message = 'File size is too large. Please upload a file smaller than 16MB.';
        } else if (error.message.includes('429')) {
            message = '⏱️ Rate limit exceeded. You\'ve made too many requests. Please wait a minute and try again.';
        } else if (error.message.includes('500')) {
            message = 'A server error occurred. Please try again later.';
        } else {
            message = error.message || `An error occurred while ${action === 'analyze' ? 'analyzing the resume' : 'calculating the match'}. Please try again.`;
        }

        showError(message);
    }

    // ==================== DISPLAY RESULTS ====================
    /**
     * Display analysis results
     */
    function displayResults(data) {
        if (!data || !data.predicted_job || !data.matches || !data.salary) {
            showError('Invalid response from server. Please try again.');
            return;
        }

        let html = `
            <h2> Analysis Results</h2>
            <div class="result-card">
                <h3> Predicted Job Category</h3>
                <div class="job-badge">${escapeHtml(data.predicted_job)}</div>
            </div>
            
            <div class="result-card">
                <h3> Top Job Matches</h3>
                <ul class="match-list">
        `;

        // Add job matches
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
                <h3> Estimated Salary</h3>
                <div class="salary-display">
                    <span class="salary-icon"></span>
                    <span class="salary-amount">${escapeHtml(data.salary)}</span>
                </div>
        `;

        // Add salary details if available
        if (data.salary_details) {
            const details = data.salary_details;
            const confidence = details.confidence || 0;
            const confidencePercent = (confidence * 100).toFixed(0);

            let confidenceClass = 'low';
            let confidenceLabel = 'Low';
            if (confidence >= 0.8) {
                confidenceClass = 'high';
                confidenceLabel = 'High';
            } else if (confidence >= 0.5) {
                confidenceClass = 'medium';
                confidenceLabel = 'Medium';
            }

            html += `
                <div class="salary-details">
                    <div class="confidence-indicator">
                        <span class="confidence-label">Confidence:</span>
                        <span class="confidence-badge ${confidenceClass}">${confidenceLabel} (${confidencePercent}%)</span>
                    </div>
                    <div class="confidence-bar">
                        <div class="confidence-fill ${confidenceClass}" style="width: ${confidencePercent}%"></div>
                    </div>
            `;

            // Add feature breakdown
            if (details.features) {
                const features = details.features;
                html += `
                    <div class="features-breakdown">
                        <h4> Extracted Features:</h4>
                        <div class="feature-grid">
                            <div class="feature-item">
                                <span class="feature-icon"></span>
                                <span class="feature-label">Experience:</span>
                                <span class="feature-value">${features.years_experience} years</span>
                            </div>
                            <div class="feature-item">
                                <span class="feature-icon"></span>
                                <span class="feature-label">Education:</span>
                                <span class="feature-value">${getEducationLabel(features.education_level)}</span>
                            </div>
                            <div class="feature-item">
                                <span class="feature-icon">⭐</span>
                                <span class="feature-label">Seniority:</span>
                                <span class="feature-value">${getSeniorityLabel(features.seniority_level)}</span>
                            </div>
                            <div class="feature-item">
                                <span class="feature-icon">️</span>
                                <span class="feature-label">Skills Count:</span>
                                <span class="feature-value">${features.skills_count}</span>
                            </div>
                        </div>
                    </div>
                `;
            }

            if (details.note) {
                html += `<p class="salary-note"><small>${escapeHtml(details.note)}</small></p>`;
            }

            html += `</div>`;
        }

        html += `</div>`;

        DOM.result.innerHTML = html;
        DOM.result.style.display = 'block';
        scrollToElement(DOM.result);
    }

    /**
     * Display JD match results
     */
    function displayJDMatch(data) {
        if (!data || typeof data.match_percentage !== 'number') {
            showError('Invalid response from server. Please try again.');
            return;
        }

        const percentage = data.match_percentage;
        const components = data.component_scores || {};
        const missingKeywords = data.missing_keywords || {};
        const keywordSuggestions = data.keyword_suggestions || [];
        const skillsBreakdown = data.skills_breakdown || {};

        let matchLevel = '';
        let matchClass = '';

        // Determine match level
        if (percentage >= 80) {
            matchLevel = 'Excellent Match!';
            matchClass = 'excellent';
        } else if (percentage >= 60) {
            matchLevel = 'Good Match!';
            matchClass = 'good';
        } else if (percentage >= 40) {
            matchLevel = 'Moderate Match';
            matchClass = 'moderate';
        } else {
            matchLevel = 'Low Match';
            matchClass = 'low';
        }

        let html = `
            <h2> JD Match Analysis</h2>
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

        // Add missing keywords section
        if (missingKeywords.critical || missingKeywords.important || missingKeywords.optional) {
            html += `
                <div class="result-card">
                    <h3> Missing Keywords</h3>
                    <p class="section-description">Add these keywords to improve your match score:</p>
            `;

            if (missingKeywords.critical && missingKeywords.critical.length > 0) {
                html += `
                    <div class="keywords-section">
                        <h4 class="keywords-title critical"> Critical (High Priority)</h4>
                        <div class="keywords-list">
                            ${missingKeywords.critical.map(kw => `
                                <span class="keyword-badge critical">${escapeHtml(kw)}</span>
                            `).join('')}
                        </div>
                    </div>
                `;
            }

            if (missingKeywords.important && missingKeywords.important.length > 0) {
                html += `
                    <div class="keywords-section">
                        <h4 class="keywords-title important"> Important (Medium Priority)</h4>
                        <div class="keywords-list">
                            ${missingKeywords.important.map(kw => `
                                <span class="keyword-badge important">${escapeHtml(kw)}</span>
                            `).join('')}
                        </div>
                    </div>
                `;
            }

            if (missingKeywords.optional && missingKeywords.optional.length > 0) {
                html += `
                    <div class="keywords-section">
                        <h4 class="keywords-title optional"> Optional (Low Priority)</h4>
                        <div class="keywords-list">
                            ${missingKeywords.optional.map(kw => `
                                <span class="keyword-badge optional">${escapeHtml(kw)}</span>
                            `).join('')}
                        </div>
                    </div>
                `;
            }

            // Add keyword suggestions
            if (keywordSuggestions.length > 0) {
                html += `
                    <div class="suggestions-box">
                        <h4> Recommendations:</h4>
                        <ul class="suggestions-list">
                            ${keywordSuggestions.map(suggestion => `
                                <li>${escapeHtml(suggestion)}</li>
                            `).join('')}
                        </ul>
                    </div>
                `;
            }

            html += `</div>`;
        }

        // Add skills breakdown section
        if (skillsBreakdown.missing_skills || skillsBreakdown.matched_skills) {
            html += `
                <div class="result-card">
                    <h3>⚡ Skills Analysis</h3>
            `;

            // Matched skills
            if (skillsBreakdown.matched_skills && Object.keys(skillsBreakdown.matched_skills).length > 0) {
                html += `
                    <div class="skills-section matched">
                        <h4>✅ Matched Skills</h4>
                        <div class="skills-categories">
                `;

                for (const [category, skills] of Object.entries(skillsBreakdown.matched_skills)) {
                    if (skills && skills.length > 0) {
                        html += `
                            <div class="skill-category">
                                <div class="category-name">${formatCategoryName(category)}</div>
                                <div class="skills-list">
                                    ${skills.map(skill => `
                                        <span class="skill-badge matched">${escapeHtml(skill)}</span>
                                    `).join('')}
                                </div>
                            </div>
                        `;
                    }
                }

                html += `
                        </div>
                    </div>
                `;
            }

            // Missing skills
            if (skillsBreakdown.missing_skills && Object.keys(skillsBreakdown.missing_skills).length > 0) {
                html += `
                    <div class="skills-section missing">
                        <h4>❌ Missing Skills</h4>
                        <p class="section-description">Consider adding these skills to your resume:</p>
                        <div class="skills-categories">
                `;

                for (const [category, skills] of Object.entries(skillsBreakdown.missing_skills)) {
                    if (skills && skills.length > 0) {
                        html += `
                            <div class="skill-category">
                                <div class="category-name">${formatCategoryName(category)}</div>
                                <div class="skills-list">
                                    ${skills.map(skill => `
                                        <span class="skill-badge missing">${escapeHtml(skill)}</span>
                                    `).join('')}
                                </div>
                            </div>
                        `;
                    }
                }

                html += `
                        </div>
                    </div>
                `;
            }

            html += `</div>`;
        }

        // Add component breakdown if available
        if (Object.keys(components).length > 0) {
            html += `
                <div class="result-card">
                    <h3> Detailed Score Breakdown</h3>
                    <div class="component-scores">
            `;

            const componentLabels = {
                'semantic': ' Semantic Similarity',
                'keyword': ' Keyword Match',
                'skills': '⚡ Skills Match',
                'context': ' Contextual Match'
            };

            for (const [key, value] of Object.entries(components)) {
                const label = componentLabels[key] || key;
                html += `
                    <div class="component-item">
                        <div class="component-header">
                            <span class="component-label">${label}</span>
                            <span class="component-value">${value}%</span>
                        </div>
                        <div class="component-bar">
                            <div class="component-bar-fill" style="width: ${value}%"></div>
                        </div>
                    </div>
                `;
            }

            html += `
                    </div>
                    <p class="breakdown-note">
                        <small>The final score is calculated using weighted average: 
                        Semantic (40%) + Keywords (30%) + Skills (20%) + Context (10%)</small>
                    </p>
                </div>
            `;
        }

        DOM.jdMatchResult.innerHTML = html;
        DOM.jdMatchResult.style.display = 'block';
        scrollToElement(DOM.jdMatchResult);
    }

    // ==================== UTILITY FUNCTIONS ====================
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
        return String(text).replace(/[&<>"']/g, m => map[m]);
    }

    /**
     * Format file size for display
     */
    function formatFileSize(bytes) {
        if (bytes === 0) return '0 Bytes';

        const k = 1024;
        const sizes = ['Bytes', 'KB', 'MB', 'GB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));

        return Math.round((bytes / Math.pow(k, i)) * 100) / 100 + ' ' + sizes[i];
    }

    /**
     * Get education level label
     */
    function getEducationLabel(level) {
        const labels = {
            0: 'Unknown',
            1: "Bachelor's",
            2: "Master's",
            3: 'PhD/Doctorate'
        };
        return labels[level] || 'Unknown';
    }

    /**
     * Get seniority level label
     */
    function getSeniorityLabel(level) {
        const labels = {
            0: 'Entry Level',
            1: 'Mid Level',
            2: 'Senior Level',
            3: 'Lead/Principal'
        };
        return labels[level] || 'Mid Level';
    }

    /**
     * Format category name for display
     */
    function formatCategoryName(category) {
        const names = {
            'programming_languages': ' Programming Languages',
            'web_frameworks': ' Web Frameworks',
            'databases': '️ Databases',
            'cloud_platforms': '☁️ Cloud Platforms',
            'devops_tools': ' DevOps Tools',
            'data_science_ml': ' Data Science & ML',
            'mobile_development': ' Mobile Development',
            'testing_frameworks': ' Testing Frameworks',
            'other_technologies': '⚙️ Other Technologies',
            'methodologies': ' Methodologies',
            'soft_skills': ' Soft Skills',
            'design_tools': ' Design Tools',
            'other_tools': '️ Other Tools'
        };
        return names[category] || category.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase());
    }

    /**
     * Scroll to element smoothly
     */
    function scrollToElement(element) {
        setTimeout(() => {
            element.scrollIntoView(CONFIG.SCROLL_OPTIONS);
        }, 100);
    }

    // ==================== START APPLICATION ====================
    // Initialize when DOM is ready
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', init);
    } else {
        init();
    }

})();