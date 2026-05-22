/**
 * QuizForge AI — Application Logic
 * =================================
 * Handles all UI interactions, API calls, and dynamic rendering.
 */

const API_BASE = window.location.origin;

// ─── DOM References ──────────────────────────────────────────────────────────

const elements = {
    // Navbar
    navbar: document.getElementById('navbar'),
    apiStatus: document.getElementById('apiStatus'),

    // Tabs
    tabText: document.getElementById('tabText'),
    tabPdf: document.getElementById('tabPdf'),
    textTab: document.getElementById('textTab'),
    pdfTab: document.getElementById('pdfTab'),

    // Inputs
    contentInput: document.getElementById('contentInput'),
    charCount: document.getElementById('charCount'),
    topicInput: document.getElementById('topicInput'),
    numQuestions: document.getElementById('numQuestions'),
    numQuestionsValue: document.getElementById('numQuestionsValue'),
    difficulty: document.getElementById('difficulty'),

    // File upload
    uploadZone: document.getElementById('uploadZone'),
    pdfInput: document.getElementById('pdfInput'),
    fileInfo: document.getElementById('fileInfo'),
    fileName: document.getElementById('fileName'),
    fileRemove: document.getElementById('fileRemove'),

    // Action
    generateBtn: document.getElementById('generateBtn'),

    // Output
    outputPanel: document.getElementById('outputPanel'),
    emptyState: document.getElementById('emptyState'),
    loadingState: document.getElementById('loadingState'),
    resultsContainer: document.getElementById('resultsContainer'),
    outputActions: document.getElementById('outputActions'),
    resultsMeta: document.getElementById('resultsMeta'),
    questionsList: document.getElementById('questionsList'),

    // Buttons
    copyJsonBtn: document.getElementById('copyJsonBtn'),
    downloadBtn: document.getElementById('downloadBtn'),

    // Toast
    toast: document.getElementById('toast'),
};

// ─── State ───────────────────────────────────────────────────────────────────

let currentMode = 'text'; // 'text' or 'pdf'
let uploadedFile = null;
let lastResult = null;

// ─── Initialization ──────────────────────────────────────────────────────────

document.addEventListener('DOMContentLoaded', () => {
    initNavbar();
    initTabs();
    initInputs();
    initFileUpload();
    initGenerate();
    initOutputActions();
    checkApiHealth();
});

// ─── Navbar Scroll ───────────────────────────────────────────────────────────

function initNavbar() {
    window.addEventListener('scroll', () => {
        elements.navbar.classList.toggle('scrolled', window.scrollY > 50);
    });
}

// ─── Tab Switching ───────────────────────────────────────────────────────────

function initTabs() {
    elements.tabText.addEventListener('click', () => switchTab('text'));
    elements.tabPdf.addEventListener('click', () => switchTab('pdf'));
}

function switchTab(tab) {
    currentMode = tab;

    elements.tabText.classList.toggle('active', tab === 'text');
    elements.tabPdf.classList.toggle('active', tab === 'pdf');
    elements.textTab.classList.toggle('active', tab === 'text');
    elements.pdfTab.classList.toggle('active', tab === 'pdf');
}

// ─── Input Handlers ──────────────────────────────────────────────────────────

function initInputs() {
    // Character counter
    elements.contentInput.addEventListener('input', () => {
        elements.charCount.textContent = elements.contentInput.value.length;
    });

    // Range slider
    elements.numQuestions.addEventListener('input', () => {
        elements.numQuestionsValue.textContent = elements.numQuestions.value;
    });
}

// ─── File Upload ─────────────────────────────────────────────────────────────

function initFileUpload() {
    const zone = elements.uploadZone;

    // Click to upload
    zone.addEventListener('click', (e) => {
        if (e.target === elements.fileRemove) return;
        elements.pdfInput.click();
    });

    // File selected
    elements.pdfInput.addEventListener('change', (e) => {
        if (e.target.files.length > 0) {
            handleFile(e.target.files[0]);
        }
    });

    // Drag & Drop
    zone.addEventListener('dragover', (e) => {
        e.preventDefault();
        zone.classList.add('dragover');
    });
    zone.addEventListener('dragleave', () => {
        zone.classList.remove('dragover');
    });
    zone.addEventListener('drop', (e) => {
        e.preventDefault();
        zone.classList.remove('dragover');
        if (e.dataTransfer.files.length > 0) {
            handleFile(e.dataTransfer.files[0]);
        }
    });

    // Remove file
    elements.fileRemove.addEventListener('click', (e) => {
        e.stopPropagation();
        clearFile();
    });
}

function handleFile(file) {
    const allowedTypes = ['.pdf', '.txt', '.md'];
    const ext = '.' + file.name.split('.').pop().toLowerCase();

    if (!allowedTypes.includes(ext)) {
        showToast('Unsupported file type. Please upload a PDF, TXT, or MD file.', 'error');
        return;
    }

    uploadedFile = file;
    elements.fileName.textContent = file.name;
    elements.fileInfo.style.display = 'flex';
}

function clearFile() {
    uploadedFile = null;
    elements.pdfInput.value = '';
    elements.fileInfo.style.display = 'none';
    elements.fileName.textContent = '';
}

// ─── Quiz Generation ─────────────────────────────────────────────────────────

function initGenerate() {
    elements.generateBtn.addEventListener('click', generateQuiz);
}

async function generateQuiz() {
    // Validation
    if (currentMode === 'text') {
        const content = elements.contentInput.value.trim();
        if (content.length < 50) {
            showToast('Please enter at least 50 characters of content.', 'error');
            return;
        }
    } else {
        if (!uploadedFile) {
            showToast('Please upload a file first.', 'error');
            return;
        }
    }

    // Get selected types
    const selectedTypes = Array.from(document.querySelectorAll('.type-checkbox:checked'))
        .map(cb => cb.value);

    if (selectedTypes.length === 0) {
        showToast('Please select at least one question type.', 'error');
        return;
    }

    // Show loading state
    showLoading();
    setButtonLoading(true);

    try {
        let result;

        if (currentMode === 'text') {
            result = await generateFromText(selectedTypes);
        } else {
            result = await generateFromPDF(selectedTypes);
        }

        lastResult = result;
        renderResults(result);
        showToast(`Generated ${result.questions.length} questions successfully!`, 'success');

    } catch (error) {
        console.error('Generation failed:', error);
        showEmpty();
        showToast(`Generation failed: ${error.message}`, 'error');
    } finally {
        setButtonLoading(false);
    }
}

async function generateFromText(selectedTypes) {
    const payload = {
        content: elements.contentInput.value.trim(),
        num_questions: parseInt(elements.numQuestions.value),
        types: selectedTypes,
        generate_explanations: true,
        strict_validation: false,
    };

    const difficulty = elements.difficulty.value;
    if (difficulty) payload.difficulty = difficulty;

    const topic = elements.topicInput.value.trim();
    if (topic) payload.topic = topic;

    const response = await fetch(`${API_BASE}/api/v1/generate-quiz`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
    });

    if (!response.ok) {
        const err = await response.json().catch(() => ({}));
        throw new Error(err.detail || err.error || `HTTP ${response.status}`);
    }

    return await response.json();
}

async function generateFromPDF(selectedTypes) {
    const formData = new FormData();
    formData.append('file', uploadedFile);
    formData.append('num_questions', elements.numQuestions.value);
    formData.append('types', selectedTypes.join(','));
    formData.append('generate_explanations', 'true');

    const difficulty = elements.difficulty.value;
    if (difficulty) formData.append('difficulty', difficulty);

    const topic = elements.topicInput.value.trim();
    if (topic) formData.append('topic', topic);

    const response = await fetch(`${API_BASE}/api/v1/upload-pdf`, {
        method: 'POST',
        body: formData,
    });

    if (!response.ok) {
        const err = await response.json().catch(() => ({}));
        throw new Error(err.detail || err.error || `HTTP ${response.status}`);
    }

    return await response.json();
}

// ─── Rendering ───────────────────────────────────────────────────────────────

function renderResults(data) {
    // Hide other states
    elements.emptyState.style.display = 'none';
    elements.loadingState.style.display = 'none';
    elements.resultsContainer.style.display = 'block';
    elements.outputActions.style.display = 'flex';

    // Render metadata
    const meta = data.metadata || {};
    elements.resultsMeta.innerHTML = `
        <div class="meta-item">
            <div class="meta-value">${data.questions.length}</div>
            <div class="meta-label">Questions</div>
        </div>
        <div class="meta-item">
            <div class="meta-value">${meta.num_chunks || '—'}</div>
            <div class="meta-label">Chunks</div>
        </div>
        <div class="meta-item">
            <div class="meta-value">${meta.num_concepts_extracted || '—'}</div>
            <div class="meta-label">Concepts</div>
        </div>
        <div class="meta-item">
            <div class="meta-value">${meta.processing_time_seconds || '—'}s</div>
            <div class="meta-label">Time</div>
        </div>
    `;

    // Render questions
    elements.questionsList.innerHTML = data.questions
        .map((q, i) => renderQuestionCard(q, i))
        .join('');
}

function renderQuestionCard(q, index) {
    const diffClass = `badge-${q.difficulty || 'medium'}`;
    const typeLabel = formatType(q.type);
    const letters = ['A', 'B', 'C', 'D', 'E', 'F'];

    let bodyHTML = '';

    // Question text
    bodyHTML += `<div class="question-text">${escapeHtml(q.question)}</div>`;

    // Options (for MCQ / AssertionReason)
    if (q.options && q.options.length > 0) {
        bodyHTML += '<div class="options-list">';
        q.options.forEach((opt, i) => {
            const isCorrect = opt === q.answer ||
                (q.answer && opt.toLowerCase().trim() === q.answer.toLowerCase().trim());
            bodyHTML += `
                <div class="option-item ${isCorrect ? 'correct' : ''}">
                    <span class="option-letter">${letters[i] || '?'}</span>
                    <span>${escapeHtml(opt)}</span>
                </div>
            `;
        });
        bodyHTML += '</div>';
    } else {
        // Show answer block for non-MCQ types
        bodyHTML += `
            <div class="answer-block">
                <span class="answer-label">Answer:</span>
                <span class="answer-text">${escapeHtml(q.answer || 'N/A')}</span>
            </div>
        `;
    }

    // Explanation
    if (q.explanation) {
        bodyHTML += `
            <div class="explanation-block">
                <div class="explanation-label">💡 Explanation</div>
                <div class="explanation-text">${escapeHtml(q.explanation)}</div>
            </div>
        `;
    }

    return `
        <div class="question-card" style="animation-delay: ${index * 0.08}s">
            <div class="question-card-header">
                <span class="question-number">${index + 1}</span>
                <div class="question-badges">
                    <span class="badge badge-type">${typeLabel}</span>
                    <span class="badge ${diffClass}">${q.difficulty || 'medium'}</span>
                </div>
            </div>
            <div class="question-card-body">
                ${bodyHTML}
            </div>
        </div>
    `;
}

function formatType(type) {
    const map = {
        'MCQ': 'MCQ',
        'TrueFalse': 'True / False',
        'FillInTheBlank': 'Fill in the Blank',
        'ShortAnswer': 'Short Answer',
        'AssertionReason': 'Assertion-Reason',
    };
    return map[type] || type || 'Unknown';
}

// ─── Loading Steps Animation ─────────────────────────────────────────────────

function showLoading() {
    elements.emptyState.style.display = 'none';
    elements.resultsContainer.style.display = 'none';
    elements.outputActions.style.display = 'none';
    elements.loadingState.style.display = 'block';

    // Animate steps
    const steps = ['step1', 'step2', 'step3', 'step4', 'step5'];
    steps.forEach(id => {
        const el = document.getElementById(id);
        el.classList.remove('active', 'done');
    });

    let currentStep = 0;
    const stepInterval = setInterval(() => {
        if (currentStep > 0) {
            document.getElementById(steps[currentStep - 1]).classList.remove('active');
            document.getElementById(steps[currentStep - 1]).classList.add('done');
        }
        if (currentStep < steps.length) {
            document.getElementById(steps[currentStep]).classList.add('active');
            currentStep++;
        } else {
            clearInterval(stepInterval);
        }
    }, 2500);

    // Store interval for cleanup
    elements.loadingState._interval = stepInterval;
}

function showEmpty() {
    elements.loadingState.style.display = 'none';
    elements.resultsContainer.style.display = 'none';
    elements.outputActions.style.display = 'none';
    elements.emptyState.style.display = 'block';

    if (elements.loadingState._interval) {
        clearInterval(elements.loadingState._interval);
    }
}

// ─── Button Loading State ────────────────────────────────────────────────────

function setButtonLoading(loading) {
    const btn = elements.generateBtn;
    const content = btn.querySelector('.btn-content');
    const spinner = btn.querySelector('.btn-loading');

    btn.disabled = loading;
    content.style.display = loading ? 'none' : 'inline-flex';
    spinner.style.display = loading ? 'inline-flex' : 'none';
}

// ─── Output Actions ──────────────────────────────────────────────────────────

function initOutputActions() {
    elements.copyJsonBtn.addEventListener('click', () => {
        if (!lastResult) return;
        const json = JSON.stringify(lastResult, null, 2);
        navigator.clipboard.writeText(json).then(() => {
            showToast('JSON copied to clipboard!', 'success');
        }).catch(() => {
            showToast('Failed to copy to clipboard.', 'error');
        });
    });

    elements.downloadBtn.addEventListener('click', () => {
        if (!lastResult) return;
        const json = JSON.stringify(lastResult, null, 2);
        const blob = new Blob([json], { type: 'application/json' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `quiz_${lastResult.topic || 'output'}_${Date.now()}.json`;
        a.click();
        URL.revokeObjectURL(url);
        showToast('Quiz downloaded!', 'success');
    });
}

// ─── API Health Check ────────────────────────────────────────────────────────

async function checkApiHealth() {
    const statusDot = elements.apiStatus.querySelector('.status-dot');
    const statusText = elements.apiStatus.querySelector('.status-text');

    try {
        const response = await fetch(`${API_BASE}/api/v1/health`, { signal: AbortSignal.timeout(5000) });
        if (response.ok) {
            statusDot.classList.add('online');
            statusDot.classList.remove('offline');
            statusText.textContent = 'API Online';
        } else {
            throw new Error('Unhealthy');
        }
    } catch {
        statusDot.classList.add('offline');
        statusDot.classList.remove('online');
        statusText.textContent = 'API Offline';
    }
}

// Health check every 30 seconds
setInterval(checkApiHealth, 30000);

// ─── Toast Notification ──────────────────────────────────────────────────────

function showToast(message, type = 'info') {
    const toast = elements.toast;
    const icon = toast.querySelector('.toast-icon');
    const msg = toast.querySelector('.toast-message');

    const icons = { success: '✅', error: '❌', info: 'ℹ️' };
    icon.textContent = icons[type] || icons.info;
    msg.textContent = message;

    toast.className = `toast ${type} show`;

    clearTimeout(toast._timeout);
    toast._timeout = setTimeout(() => {
        toast.classList.remove('show');
    }, 4000);
}

// ─── Utilities ───────────────────────────────────────────────────────────────

function escapeHtml(text) {
    if (!text) return '';
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}
