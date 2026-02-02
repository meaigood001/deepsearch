// API Configuration
const API_BASE = window.location.origin;

// SVG Icons (Lucide)
const ICONS = {
    view: '<svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M2 12s3-7 10-7 10 7 10 7-3 7-10 7-10-7-10-7Z"></path><circle cx="12" cy="12" r="3"></circle></svg>',
    delete: '<svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M3 6h18"></path><path d="M19 6v14c0 1-1 2-2 2H7c-1 0-2-1-2-2V6"></path><path d="M8 6V4c0-1 1-2 2-2h4c1 0 2 1 2 2v2"></path></svg>',
    search: '<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="11" cy="11" r="8"></circle><path d="m21 21-4.3-4.3"></path></svg>',
    globe: '<svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"></circle><path d="M12 2a14.5 14.5 0 0 0 0 20 14.5 14.5 0 0 0 0-20"></path><path d="M2 12h20"></path></svg>',
    target: '<svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"></circle><circle cx="12" cy="12" r="6"></circle><circle cx="12" cy="12" r="2"></circle></svg>',
    list: '<svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="8" x2="21" y1="6" y2="6"></line><line x1="8" x2="21" y1="12" y2="12"></line><line x1="8" x2="21" y1="18" y2="18"></line><line x1="3" x2="3.01" y1="6" y2="6"></line><line x1="3" x2="3.01" y1="12" y2="12"></line><line x1="3" x2="3.01" y1="18" y2="18"></line></svg>',
    zap: '<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"></polygon></svg>',
    fileText: '<svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M14.5 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V7.5L14.5 2z"></path><polyline points="14 2 14 8 20 8"></polyline><line x1="16" x2="8" y1="13" y2="13"></line><line x1="16" x2="8" y1="17" y2="17"></line><line x1="10" x2="8" y1="9" y2="9"></line></svg>',
    check: '<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="20 6 9 17 4 12"></polyline></svg>',
    alertTriangle: '<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="m21.73 18-8-14a2 2 0 0 0-3.48 0l-8 14A2 2 0 0 0 4 21h16a2 2 0 0 0 1.73-3Z"></path><line x1="12" x2="12" y1="9" y2="13"></line><line x1="12" x2="12.01" y1="17" y2="17"></line></svg>',
    download: '<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path><polyline points="7 10 12 15 17 10"></polyline><line x1="12" x2="12" y1="15" y2="3"></line></svg>',
    link: '<svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M10 13a5 5 0 0 0 7.54.54l3-3a5 5 0 0 0-7.07-7.07l-1.72 1.71"></path><path d="M14 11a5 5 0 0 0-7.54-.54l-3 3a5 5 0 0 0 7.07 7.07l1.71-1.71"></path></svg>',
    user: '<svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M19 21v-2a4 4 0 0 0-4-4H9a4 4 0 0 0-4 4v2"></path><circle cx="12" cy="7" r="4"></circle></svg>',
    chevronDown: '<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="6 9 12 15 18 9"></polyline></svg>',
    logout: '<svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M9 21H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h4"></path><polyline points="16 17 21 12 16 7"></polyline><line x1="21" y1="12" x2="9" y2="12"></line></svg>',
    signIn: '<svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M15 3h4a2 2 0 0 1 2 2v14a2 2 0 0 1-2 2h-4"></path><polyline points="10 17 15 12 10 7"></polyline><line x1="15" y1="12" x2="3" y2="12"></line></svg>',
    signUp: '<svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M16 21v-2a4 4 0 0 0-4-4H6a4 4 0 0 0-4 4v2"></path><circle cx="9" cy="7" r="4"></circle><line x1="19" y1="8" x2="19" y2="14"></line><line x1="22" y1="11" x2="16" y2="11"></line></svg>'
};

// State Management
const state = {
    currentResearch: null,
    isResearching: false,
    eventSource: null,
    clarificationQuestions: [],
    config: {
        max_iterations: 3,
        lang: 'en',
        char_limits: {
            background: 300,
            keyword_summary: 500,
            final_report: 2000
        }
    },
    currentUser: null
};

// DOM Elements
const elements = {
    healthStatus: document.getElementById('healthStatus'),
    queryInput: document.getElementById('queryInput'),
    startResearch: document.getElementById('startResearch'),
    resetResearch: document.getElementById('resetResearch'),
    progressSection: document.getElementById('progressSection'),
    progressFill: document.getElementById('progressFill'),
    statusMessage: document.getElementById('statusMessage'),
    resultsDisplay: document.getElementById('resultsDisplay'),
    clarificationSection: document.getElementById('clarificationSection'),
    clarificationQuestions: document.getElementById('clarificationQuestions'),
    clarificationForm: document.getElementById('clarificationForm'),
    historyList: document.getElementById('historyList'),
    refreshHistory: document.getElementById('refreshHistory'),
    toggleSidebar: document.getElementById('toggleSidebar'),
    maxIterations: document.getElementById('maxIterations'),
    maxIterationsValue: document.getElementById('maxIterationsValue'),
    language: document.getElementById('language'),
    limitBackground: document.getElementById('limitBackground'),
    limitKeyword: document.getElementById('limitKeyword'),
    limitFinal: document.getElementById('limitFinal'),
    // Auth elements
    authModal: document.getElementById('authModal'),
    authModalOverlay: document.getElementById('authModalOverlay'),
    authModalClose: document.getElementById('authModalClose'),
    loginForm: document.getElementById('loginForm'),
    registerForm: document.getElementById('registerForm'),
    loginFormSubmit: document.getElementById('loginFormSubmit'),
    registerFormSubmit: document.getElementById('registerFormSubmit'),
    loginUsername: document.getElementById('loginUsername'),
    loginPassword: document.getElementById('loginPassword'),
    registerUsername: document.getElementById('registerUsername'),
    registerEmail: document.getElementById('registerEmail'),
    registerPassword: document.getElementById('registerPassword'),
    loginError: document.getElementById('loginError'),
    registerError: document.getElementById('registerError'),
    showRegister: document.getElementById('showRegister'),
    showLogin: document.getElementById('showLogin'),
    userMenu: document.getElementById('userMenu'),
    userMenuTrigger: document.getElementById('userMenuTrigger'),
    userMenuDropdown: document.getElementById('userMenuDropdown'),
    userMenuName: document.getElementById('userMenuName'),
    userDropdownName: document.getElementById('userDropdownName'),
    userDropdownEmail: document.getElementById('userDropdownEmail'),
    logoutBtn: document.getElementById('logoutBtn')
};


// Initialize Application
async function init() {
    // Initially disable protected features until auth status is confirmed
    disableProtectedFeatures();
    
    // Configure marked.js for markdown rendering
    marked.setOptions({
        highlight: function(code, lang) {
            if (lang && hljs.getLanguage(lang)) {
                try {
                    return hljs.highlight(code, { language: lang }).value;
                } catch (err) {
                    console.error('Highlight error:', err);
                }
            }
            return hljs.highlightAuto(code).value;
        },
        breaks: true,
        gfm: true,
        headerIds: true,
        mangle: false
    });
    
    await checkHealth();
    await checkAuthStatus();
    setupEventListeners();
    updateConfigFromInputs();
}

// Check authentication status
async function checkAuthStatus() {
    try {
        const response = await fetch(`${API_BASE}/api/auth/me`);
        
        if (response.status === 401) {
            // Explicitly unauthorized
            state.currentUser = null;
            showAuthModal(true); // Force modal to stay open
            disableProtectedFeatures();
            return;
        }
        
        const data = await response.json();
        
        if (data.success) {
            state.currentUser = data.user;
            showUserMenu(data.user);
            enableProtectedFeatures();
            await loadHistory(); // Load user-specific history
        } else {
            state.currentUser = null;
            showAuthModal(true); // Force modal to stay open
            disableProtectedFeatures();
        }
    } catch (error) {
        console.error('Auth status check failed:', error);
        state.currentUser = null;
        showAuthModal(true); // Force modal to stay open
        disableProtectedFeatures();
    }
}

// Disable protected features when not authenticated
function disableProtectedFeatures() {
    elements.startResearch.disabled = true;
    elements.startResearch.style.opacity = '0.5';
    elements.startResearch.style.cursor = 'not-allowed';
    elements.queryInput.disabled = true;
    elements.queryInput.placeholder = 'Please sign in to start research...';
    elements.queryInput.style.backgroundColor = 'var(--bg-base)';
    elements.historyList.innerHTML = '<div class="info-box">Please sign in to view your research history.</div>';
}

// Enable protected features when authenticated
function enableProtectedFeatures() {
    elements.startResearch.disabled = false;
    elements.startResearch.style.opacity = '1';
    elements.startResearch.style.cursor = 'pointer';
    elements.queryInput.disabled = false;
    elements.queryInput.placeholder = 'e.g., What are the latest developments in AI safety?';
    elements.queryInput.style.backgroundColor = '';
}

// Show authentication modal and prevent access to main UI
function showAuthModal(force = false) {
    elements.authModal.style.display = 'flex';
    document.body.style.overflow = 'hidden'; // Prevent scrolling when modal is open
    
    // Disable main content when auth modal is shown
    const mainContent = document.querySelector('.main-content');
    if (mainContent) {
        mainContent.style.pointerEvents = 'none';
        mainContent.style.opacity = '0.3';
    }
    
    // Hide close button to force login when not authenticated
    if (elements.authModalClose) {
        if (force || !state.currentUser) {
            elements.authModalClose.style.display = 'none';
        } else {
            elements.authModalClose.style.display = 'block';
        }
    }
}

// Hide authentication modal and restore access to main UI
function hideAuthModal() {
    elements.authModal.style.display = 'none';
    document.body.style.overflow = ''; // Restore scrolling
    
    // Re-enable main content
    const mainContent = document.querySelector('.main-content');
    if (mainContent) {
        mainContent.style.pointerEvents = 'auto';
        mainContent.style.opacity = '1';
    }
    
    // Show close button again
    if (elements.authModalClose) {
        elements.authModalClose.style.display = 'block';
    }
    
    // Clear forms
    if (elements.loginFormSubmit) elements.loginFormSubmit.reset();
    if (elements.registerFormSubmit) elements.registerFormSubmit.reset();
    if (elements.loginError) elements.loginError.textContent = '';
    if (elements.registerError) elements.registerError.textContent = '';
}

// Show login form
function showLoginForm() {
    elements.loginForm.style.display = 'block';
    elements.registerForm.style.display = 'none';
    elements.loginError.textContent = '';
    elements.registerError.textContent = '';
}

// Show register form
function showRegisterForm() {
    elements.registerForm.style.display = 'block';
    elements.loginForm.style.display = 'none';
    elements.loginError.textContent = '';
    elements.registerError.textContent = '';
}

// Show user menu
function showUserMenu(user) {
    elements.userMenu.style.display = 'flex';
    elements.userMenuName.textContent = user.username;
    elements.userDropdownName.textContent = user.username;
    elements.userDropdownEmail.textContent = user.email;
}

// Hide user menu
function hideUserMenu() {
    elements.userMenu.style.display = 'none';
    elements.userMenuName.textContent = '';
    elements.userDropdownName.textContent = '';
    elements.userDropdownEmail.textContent = '';
}

// Login function
async function login(username, password) {
    try {
        const response = await fetch(`${API_BASE}/api/auth/login`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ username, password })
        });
        
        const data = await response.json();
        
        if (data.success) {
            state.currentUser = data.user;
            showUserMenu(data.user);
            hideAuthModal(); // Hide modal on successful login
            enableProtectedFeatures();
            await loadHistory(); // Reload history for logged-in user
            showToast('Login successful!', 'success');
            return true;
        } else {
            elements.loginError.textContent = data.error || 'Login failed';
            return false;
        }
    } catch (error) {
        console.error('Login error:', error);
        elements.loginError.textContent = 'Network error occurred';
        return false;
    }
}

// Register function
async function register(username, email, password) {
    try {
        const response = await fetch(`${API_BASE}/api/auth/register`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ username, email, password })
        });
        
        const data = await response.json();
        
        if (data.success) {
            state.currentUser = data.user;
            showUserMenu(data.user);
            hideAuthModal(); // Hide modal on successful registration
            enableProtectedFeatures();
            await loadHistory(); // Load history for new user
            showToast('Registration successful!', 'success');
            return true;
        } else {
            elements.registerError.textContent = data.error || 'Registration failed';
            return false;
        }
    } catch (error) {
        console.error('Registration error:', error);
        elements.registerError.textContent = 'Network error occurred';
        return false;
    }
}

// Logout function
async function logout() {
    try {
        await fetch(`${API_BASE}/api/auth/logout`, {
            method: 'POST'
        });
        
        state.currentUser = null;
        hideUserMenu();
        disableProtectedFeatures();
        showAuthModal(true); // Show auth modal after logout and force it to stay open
        elements.resultsDisplay.innerHTML = '';
        showToast('Logged out successfully', 'info');
    } catch (error) {
        console.error('Logout error:', error);
        showToast('Logout failed', 'error');
    }
}

// Health Check
async function checkHealth() {
    try {
        const response = await fetch(`${API_BASE}/api/health`);
        const data = await response.json();
        
        elements.healthStatus.classList.toggle('healthy', data.status === 'healthy');
        elements.healthStatus.classList.toggle('unhealthy', data.status !== 'healthy');
        
        if (data.status === 'healthy') {
            elements.healthStatus.querySelector('.status-text').textContent = 'Connected';
        } else {
            elements.healthStatus.querySelector('.status-text').textContent = 'Disconnected';
        }
        
        if (!data.openai_configured) {
            showToast('OPENAI_API_KEY not configured', 'error');
        }
        
        if (!data.tavily_configured) {
            showToast('TAVILY_API_KEY not configured', 'warning');
        }
    } catch (error) {
        console.error('Health check failed:', error);
        elements.healthStatus.classList.add('unhealthy');
        elements.healthStatus.querySelector('.status-text').textContent = 'Error';
    }
}

// Load Research History
async function loadHistory() {
    if (!state.currentUser) {
        elements.historyList.innerHTML = '<div class="info-box">Please sign in to view your research history.</div>';
        return;
    }
    
    try {
        const response = await fetch(`${API_BASE}/api/history`);
        if (response.status === 401) {
            showToast('Session expired. Please sign in again.', 'error');
            logout();
            return;
        }
        
        const data = await response.json();
        
        if (data.success && data.data.length > 0) {
            renderHistory(data.data);
        } else {
            elements.historyList.innerHTML = '<div class="info-box">No research history yet.</div>';
        }
    } catch (error) {
        console.error('Failed to load history:', error);
        if (error.message && error.message.includes('401')) {
            showToast('Session expired. Please sign in again.', 'error');
            logout();
        } else {
            elements.historyList.innerHTML = '<div class="info-box">Failed to load history.</div>';
        }
    }
}

// Render History
function renderHistory(history) {
    elements.historyList.innerHTML = '';
    
    history.forEach(item => {
        const historyItem = document.createElement('div');
        historyItem.className = 'history-item';
        historyItem.dataset.id = item.id;
        
        const query = document.createElement('div');
        query.className = 'history-item-query';
        query.textContent = item.query;
        
        const date = document.createElement('div');
        date.className = 'history-item-date';
        date.textContent = formatDate(item.created_at);
        
        const actions = document.createElement('div');
        actions.className = 'history-item-actions';
        
        const viewBtn = document.createElement('button');
        viewBtn.className = 'btn-view';
        viewBtn.innerHTML = `${ICONS.view} View`;
        viewBtn.onclick = () => viewResearch(item.id);
        
        const deleteBtn = document.createElement('button');
        deleteBtn.className = 'btn-delete';
        deleteBtn.innerHTML = `${ICONS.delete} Delete`;
        deleteBtn.onclick = () => deleteResearch(item.id);
        
        actions.appendChild(viewBtn);
        actions.appendChild(deleteBtn);
        
        historyItem.appendChild(query);
        historyItem.appendChild(date);
        historyItem.appendChild(actions);
        
        elements.historyList.appendChild(historyItem);
    });
}

// View Research
async function viewResearch(id) {
    if (!state.currentUser) {
        showToast('Please sign in to view research.', 'error');
        showAuthModal(true); // Force modal to stay open
        return;
    }
    
    try {
        const response = await fetch(`${API_BASE}/api/research/${id}`);
        if (response.status === 401) {
            showToast('Session expired. Please sign in again.', 'error');
            logout();
            return;
        }
        
        const data = await response.json();
        
        if (data.success) {
            elements.queryInput.value = data.data.query;
            state.currentResearch = data.data;
            displayResults(data.data.result);
            showToast('Research loaded from history', 'success');
        } else {
            showToast(data.error || 'Failed to load research', 'error');
        }
    } catch (error) {
        console.error('Failed to view research:', error);
        if (error.message && error.message.includes('401')) {
            showToast('Session expired. Please sign in again.', 'error');
            logout();
        } else {
            showToast('Failed to load research', 'error');
        }
    }
}

// Delete Research
async function deleteResearch(id) {
    if (!state.currentUser) {
        showToast('Please sign in to delete research.', 'error');
        showAuthModal(true); // Force modal to stay open
        return;
    }
    
    if (!confirm('Are you sure you want to delete this research?')) return;
    
    try {
        const response = await fetch(`${API_BASE}/api/research/${id}`, {
            method: 'DELETE'
        });
        
        if (response.status === 401) {
            showToast('Session expired. Please sign in again.', 'error');
            logout();
            return;
        }
        
        const data = await response.json();
        
        if (data.success) {
            await loadHistory();
            showToast('Research deleted successfully', 'success');
        } else {
            showToast(data.error || 'Failed to delete research', 'error');
        }
    } catch (error) {
        console.error('Failed to delete research:', error);
        if (error.message && error.message.includes('401')) {
            showToast('Session expired. Please sign in again.', 'error');
            logout();
        } else {
            showToast('Failed to delete research', 'error');
        }
    }
}

// Start Research
async function startResearch(userAnswers = null) {
    if (!state.currentUser) {
        showToast('Please sign in to start research.', 'error');
        showAuthModal(true); // Force modal to stay open
        return;
    }
    
    const query = elements.queryInput.value.trim();
    
    if (!query) {
        showToast('Please enter a research question', 'error');
        return;
    }
    
    state.isResearching = true;
    elements.startResearch.style.display = 'none';
    elements.resetResearch.style.display = 'inline-flex';
    elements.progressSection.style.display = 'block';
    elements.resultsDisplay.innerHTML = '';
    
    updateProgress(0, 'Initializing research...');
    
    try {
        const response = await fetch(`${API_BASE}/api/research/stream`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                query,
                config: state.config,
                user_answers: userAnswers
            })
        });
        
        if (!response.ok) {
            if (response.status === 401) {
                showToast('Session expired. Please sign in again.', 'error');
                logout();
                return;
            }
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let buffer = '';
        
        while (true) {
            const { value, done } = await reader.read();
            
            if (done) break;
            
            buffer += decoder.decode(value, { stream: true });
            const lines = buffer.split('\n');
            buffer = lines.pop();
            
            for (const line of lines) {
                if (line.startsWith('data: ')) {
                    const data = JSON.parse(line.slice(6));
                    handleStreamUpdate(data);
                }
            }
        }
        
        state.isResearching = false;
        updateProgress(100, 'Research completed!');
        showToast('Research completed successfully', 'success');
        
    } catch (error) {
        console.error('Research error:', error);
        if (error.message.includes('401')) {
            showToast('Session expired. Please sign in again.', 'error');
            logout();
        } else {
            showToast(`Research failed: ${error.message}`, 'error');
        }
        resetResearch();
    }
}

// Handle Stream Updates
function handleStreamUpdate(update) {
    const { node, state: updateState, output } = update;
    
    switch (node) {
        case 'clarify_query':
            if (output?.clarification_needed) {
                handleClarificationNeeded(updateState);
            }
            updateProgress(10, 'Analyzing query...');
            break;
            
        case 'background_search':
            updateProgress(20, 'Performing background search...');
            if (updateState?.llm_outputs?.background_search) {
                displayBackgroundSearch(updateState.llm_outputs.background_search);
            }
            break;
            
        case 'generate_keywords':
            updateProgress(30, 'Generating keywords...');
            if (updateState?.llm_outputs) {
                const iteration = updateState.iteration - 1;
                const keywordData = updateState.llm_outputs[`generate_keywords_iteration_${iteration}`];
                if (keywordData) {
                    displayKeywords(keywordData, iteration);
                }
            }
            break;
            
        case 'multi_search':
            updateProgress(50, 'Searching and summarizing...');
            if (updateState?.llm_outputs) {
                const iteration = updateState.iteration - 1;
                const searchData = updateState.llm_outputs[`multi_search_iteration_${iteration}`];
                if (searchData) {
                    displaySearchResults(searchData, iteration);
                }
            }
            break;
            
        case 'check_gaps':
            updateProgress(70, 'Analyzing research gaps...');
            if (updateState?.llm_outputs) {
                const iteration = updateState.iteration - 1;
                const gapData = updateState.llm_outputs[`check_gaps_iteration_${iteration}`];
                if (gapData) {
                    displayGapAnalysis(gapData, iteration);
                }
            }
            break;
            
        case 'synthesize':
            updateProgress(90, 'Synthesizing final report...');
            if (updateState?.llm_outputs?.synthesize) {
                displayFinalReport(updateState.llm_outputs.synthesize);
            }
            break;
            
        case 'complete':
            updateProgress(100, 'Research completed!');
            state.currentResearch = updateState;
            break;
            
        case 'saved':
            console.log('Research saved with ID:', update.research_id);
            loadHistory();
            break;
            
        case 'error':
            showToast(`Error: ${update.error}`, 'error');
            resetResearch();
            break;
    }
}

// Handle Clarification Needed
function handleClarificationNeeded(researchState) {
    state.isResearching = false;
    elements.startResearch.style.display = 'none';
    elements.resetResearch.style.display = 'inline-flex';
    
    const questions = researchState.clarification_questions || [];
    state.clarificationQuestions = questions;
    
    elements.clarificationQuestions.innerHTML = questions.map((q, idx) => `
        <div class="form-group">
            <label for="clarification_${idx}">Q${idx + 1}: ${escapeHtml(q)}</label>
            <input type="text" id="clarification_${idx}" name="${idx + 1}" required>
        </div>
    `).join('');
    
    elements.clarificationSection.style.display = 'block';
}

// Display Background Search
function displayBackgroundSearch(data) {
    const section = createResultSection(`${ICONS.globe} Background Search`, renderMarkdown(data.summary));
    
    if (data.sources && data.sources.length > 0) {
        const sources = document.createElement('div');
        sources.className = 'sources-list';
        sources.innerHTML = '<strong>Sources:</strong><ul>' + 
            data.sources.map(s => `<li><a href="${s.url}" target="_blank" rel="noopener noreferrer">${ICONS.link} ${escapeHtml(s.title || s.url)}</a></li>`).join('') +
            '</ul>';
        section.appendChild(sources);
    }
    
    elements.resultsDisplay.appendChild(section);
}

// Display Keywords
function displayKeywords(data, iteration) {
    const section = document.createElement('div');
    section.className = 'result-section';
    section.innerHTML = `<h3>${ICONS.target} Keywords (Iteration ${iteration + 1})</h3>`;
    
    const keywordsContainer = document.createElement('div');
    keywordsContainer.style.marginTop = '1rem';
    
    const keywords = data.parsed_keywords || [];
    keywords.forEach((keyword, idx) => {
        setTimeout(() => {
            const tag = document.createElement('span');
            tag.className = 'keyword-tag';
            tag.textContent = keyword;
            keywordsContainer.appendChild(tag);
        }, idx * 100);
    });
    
    section.appendChild(keywordsContainer);
    elements.resultsDisplay.appendChild(section);
}

// Display Search Results
function displaySearchResults(data, iteration) {
    const section = document.createElement('div');
    section.className = 'result-section';
    section.innerHTML = `<h3>${ICONS.list} Search Results (Iteration ${iteration + 1})</h3>`;
    
    data.forEach(item => {
        const expandable = createExpandableItem(
            `${ICONS.search} ${item.keyword}`,
            renderMarkdown(item.summary)
        );
        section.appendChild(expandable);
    });
    
    elements.resultsDisplay.appendChild(section);
}

// Display Gap Analysis
function displayGapAnalysis(data, iteration) {
    const gapsFound = data.gaps_found;
    const statusIcon = gapsFound ? ICONS.alertTriangle : ICONS.check;
    const status = gapsFound ? 'Gaps detected' : 'No gaps found';
    
    const analysisHtml = `
        <div class="markdown-content">
            <p><strong>Status:</strong> ${statusIcon} ${status}</p>
            <div style="margin-top: 1rem;">
                <strong>Analysis:</strong>
                ${renderMarkdown(data.raw_output || '')}
            </div>
        </div>
    `;
    
    const section = createResultSection(
        `${ICONS.zap} Gap Analysis (Iteration ${iteration + 1})`,
        analysisHtml
    );
    
    elements.resultsDisplay.appendChild(section);
}

// Display Final Report
function displayFinalReport(data) {
    const section = createResultSection(`${ICONS.fileText} Final Report`, renderMarkdown(data.report));
    
    if (data.sources && data.sources.length > 0) {
        const sources = document.createElement('div');
        sources.className = 'sources-list';
        sources.style.marginTop = '1rem';
        sources.innerHTML = '<strong>Sources:</strong><ul>' + 
            data.sources.map(s => `<li><a href="${s.url}" target="_blank" rel="noopener noreferrer">${ICONS.link} ${escapeHtml(s.title || s.url)}</a></li>`).join('') +
            '</ul>';
        section.appendChild(sources);
    }
    
    const downloadBtn = document.createElement('button');
    downloadBtn.className = 'btn btn-primary';
    downloadBtn.style.marginTop = '1rem';
    downloadBtn.innerHTML = `<span class="btn-icon">${ICONS.download}</span> Download Report`;
    downloadBtn.onclick = () => downloadReport(data.report);
    section.appendChild(downloadBtn);
    
    elements.resultsDisplay.appendChild(section);
}

// Display Results
function displayResults(result) {
    elements.resultsDisplay.innerHTML = '';
    
    if (result.background) {
        const bg = result.llm_outputs?.background_search;
        if (bg) displayBackgroundSearch(bg);
    }
    
    if (result.keyword_history) {
        const section = createResultSection(`${ICONS.target} Keywords Used`, '');
        const keywordsContainer = document.createElement('div');
        keywordsContainer.style.marginTop = '1rem';
        result.keyword_history.forEach(kw => {
            const tag = document.createElement('span');
            tag.className = 'keyword-tag';
            tag.textContent = kw;
            keywordsContainer.appendChild(tag);
        });
        section.appendChild(keywordsContainer);
        elements.resultsDisplay.appendChild(section);
    }
    
    if (result.llm_outputs) {
        Object.keys(result.llm_outputs).forEach(key => {
            if (key.startsWith('multi_search_iteration_')) {
                const iteration = key.split('_').pop();
                displaySearchResults(result.llm_outputs[key], parseInt(iteration));
            }
        });
    }
    
    if (result.final_report) {
        const synData = result.llm_outputs?.synthesize;
        if (synData) displayFinalReport(synData);
    }
}

// Create Result Section
function createResultSection(title, content) {
    const section = document.createElement('div');
    section.className = 'result-section';
    
    const titleEl = document.createElement('h3');
    titleEl.innerHTML = title;
    
    const contentEl = document.createElement('div');
    contentEl.className = 'markdown-content';
    contentEl.innerHTML = content;
    
    section.appendChild(titleEl);
    section.appendChild(contentEl);
    
    return section;
}

// Create Expandable Item
function createExpandableItem(title, content) {
    const item = document.createElement('div');
    item.className = 'expandable-item';
    
    const header = document.createElement('div');
    header.className = 'expandable-header collapsed';
    header.innerHTML = title;
    
    const contentDiv = document.createElement('div');
    contentDiv.className = 'expandable-content markdown-content';
    contentDiv.innerHTML = content;
    contentDiv.style.display = 'none';
    
    header.onclick = () => {
        const isCollapsed = contentDiv.style.display === 'none';
        contentDiv.style.display = isCollapsed ? 'block' : 'none';
        header.classList.toggle('collapsed', !isCollapsed);
    };
    
    item.appendChild(header);
    item.appendChild(contentDiv);
    
    return item;
}

// Update Progress
function updateProgress(percent, message) {
    elements.progressFill.style.width = `${percent}%`;
    elements.statusMessage.textContent = message;
}

// Reset Research
function resetResearch() {
    state.isResearching = false;
    state.currentResearch = null;
    elements.startResearch.style.display = 'inline-flex';
    elements.resetResearch.style.display = 'none';
    elements.progressSection.style.display = 'none';
    elements.clarificationSection.style.display = 'none';
    elements.resultsDisplay.innerHTML = '';
    elements.queryInput.value = '';
}

// Download Report
function downloadReport(content) {
    const blob = new Blob([content], { type: 'text/markdown' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'research_report.md';
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
}

// Show Toast
function showToast(message, type = 'info') {
    const toast = document.createElement('div');
    toast.className = `toast ${type}`;
    toast.textContent = message;
    
    document.getElementById('toastContainer').appendChild(toast);
    
    setTimeout(() => {
        toast.style.animation = 'slideOutRight 0.3s ease-out';
        setTimeout(() => toast.remove(), 300);
    }, 5000);
}

// Update Config From Inputs
function updateConfigFromInputs() {
    state.config = {
        max_iterations: parseInt(elements.maxIterations.value),
        lang: elements.language.value,
        char_limits: {
            background: parseInt(elements.limitBackground.value),
            keyword_summary: parseInt(elements.limitKeyword.value),
            final_report: parseInt(elements.limitFinal.value)
        }
    };
}

// Render Markdown Safely
function renderMarkdown(text) {
    if (!text) return '';
    const rawHtml = marked.parse(text);
    return DOMPurify.sanitize(rawHtml);
}

// Setup Event Listeners
function setupEventListeners() {
    // Start Research Button
    elements.startResearch.addEventListener('click', () => startResearch());
    
    // Reset Research Button
    elements.resetResearch.addEventListener('click', resetResearch);
    
    // Clarification Form
    elements.clarificationForm.addEventListener('submit', (e) => {
        e.preventDefault();
        const formData = new FormData(e.target);
        const answers = {};
        for (const [key, value] of formData.entries()) {
            answers[key] = value;
        }
        elements.clarificationSection.style.display = 'none';
        startResearch(answers);
    });
    
    // Config Updates
    elements.maxIterations.addEventListener('input', (e) => {
        elements.maxIterationsValue.textContent = e.target.value;
        updateConfigFromInputs();
    });
    
    elements.language.addEventListener('change', updateConfigFromInputs);
    elements.limitBackground.addEventListener('change', updateConfigFromInputs);
    elements.limitKeyword.addEventListener('change', updateConfigFromInputs);
    elements.limitFinal.addEventListener('change', updateConfigFromInputs);
    
    // Refresh History
    elements.refreshHistory.addEventListener('click', loadHistory);
    
    // Toggle Sidebar
    elements.toggleSidebar.addEventListener('click', () => {
        const sidebar = document.getElementById('sidebar');
        const mainContent = document.querySelector('.main-content');
        sidebar.classList.toggle('collapsed');
        mainContent.classList.toggle('sidebar-collapsed');
    });
    
    // Authentication Modal Event Listeners
    elements.authModalOverlay.addEventListener('click', (e) => {
        // Don't close modal when not authenticated
        if (!state.currentUser) {
            showToast('Please sign in to continue.', 'info');
            return;
        }
        hideAuthModal();
    });
    
    // Only allow closing the modal if the user is authenticated
    elements.authModalClose.addEventListener('click', () => {
        if (state.currentUser) {
            hideAuthModal();
        } else {
            // Don't close the modal if not authenticated
            showToast('Please sign in to continue.', 'info');
        }
    });
    
    elements.showRegister.addEventListener('click', showRegisterForm);
    elements.showLogin.addEventListener('click', showLoginForm);
    
    elements.loginFormSubmit.addEventListener('submit', async (e) => {
        e.preventDefault();
        const username = elements.loginUsername.value;
        const password = elements.loginPassword.value;
        await login(username, password);
    });
    
    elements.registerFormSubmit.addEventListener('submit', async (e) => {
        e.preventDefault();
        const username = elements.registerUsername.value;
        const email = elements.registerEmail.value;
        const password = elements.registerPassword.value;
        await register(username, email, password);
    });
    
    // User Menu Event Listeners
    elements.userMenuTrigger.addEventListener('click', () => {
        elements.userMenuDropdown.classList.toggle('show');
    });
    
    elements.logoutBtn.addEventListener('click', logout);
    
    // Close dropdown when clicking outside
    document.addEventListener('click', (e) => {
        if (!elements.userMenu.contains(e.target)) {
            elements.userMenuDropdown.classList.remove('show');
        }
    });
}

// Utility Functions
function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

function formatDate(dateString) {
    const date = new Date(dateString);
    return date.toLocaleDateString() + ' ' + date.toLocaleTimeString();
}

// Initialize on DOM ready
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
} else {
    init();
}
