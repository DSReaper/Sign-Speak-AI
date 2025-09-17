class AISignLanguageDetection {
    constructor() {
        this.aiCameraFeed = document.getElementById('aiCameraFeed');
        this.cameraSection = document.getElementById('cameraSection');
        this.cameraLoading = document.getElementById('cameraLoading');
        this.cameraError = document.getElementById('cameraError');
        this.detectedPhrase = document.getElementById('detectedPhrase');
        this.bufferStatus = document.getElementById('bufferStatus');
        this.confidenceStatus = document.getElementById('confidenceStatus');
        this.aiStatus = document.getElementById('aiStatus');
        
        this.isExpanded = false;
        this.isAIActive = false;
        this.currentMode = 'basic';
        this.flaskUrl = 'http://localhost:8001/ai'; // Node.js proxy to Flask AI server
        this.statusUpdateInterval = null;
        
        this.initializeEventListeners();
        this.startAICamera();
    }

    initializeEventListeners() {
        // Mode switching
        document.getElementById('basicDetect').addEventListener('click', () => {
            this.switchMode('basic');
        });
        
        document.getElementById('advancedDetect').addEventListener('click', () => {
            this.switchMode('advanced');
        });
        
        // AI Controls
        document.getElementById('toggleHands').addEventListener('click', () => {
            this.toggleHandDetection();
        });
        
        document.getElementById('resetBuffer').addEventListener('click', () => {
            this.resetBuffer();
        });
        
        // Expand/contract camera view
        document.getElementById('expandBtn').addEventListener('click', () => {
            this.toggleExpanded();
        });
        
        // Audio playback
        document.getElementById('playBtn').addEventListener('click', () => {
            this.playDetectedPhrase();
        });
        
        // Star/favorite functionality
        document.getElementById('starBtn').addEventListener('click', () => {
            this.toggleStar();
        });
        
        // Retry camera access
        document.getElementById('retryCamera').addEventListener('click', () => {
            this.startAICamera();
        });
        
        // Handle window resize
        window.addEventListener('resize', () => {
            this.handleResize();
        });
        
        // Handle escape key for fullscreen
        document.addEventListener('keydown', (e) => {
            if (e.key === 'Escape' && this.isExpanded) {
                this.toggleExpanded();
            }
        });

        // Modal close
        document.getElementById('closeModal').addEventListener('click', () => {
            const modal = document.getElementById('responseModal');
            const textarea = document.getElementById('userResponse');
            
            modal.style.display = 'none';
            textarea.value = '';
        });
    }

    async startAICamera() {
        try {
            this.showLoading(true);
            this.hideError();
            
            console.log('Starting AI camera...');
            
            // Start the Flask AI camera
            const response = await fetch(`${this.flaskUrl}/start_camera`, { 
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                }
            });
            
            if (!response.ok) {
                throw new Error(`Flask server responded with status: ${response.status}`);
            }
            
            const data = await response.json();
            
            if (data.status === 'success') {
                // Set the video feed source
                this.aiCameraFeed.src = `${this.flaskUrl}/video_feed?t=${Date.now()}`;
                this.aiCameraFeed.style.display = 'block';
                this.showLoading(false);
                this.isAIActive = true;
                
                // Start status updates
                this.startStatusUpdates();
                
                console.log('AI camera started successfully');
            } else {
                throw new Error(data.message || 'Failed to start AI camera');
            }
            
        } catch (error) {
            console.error('AI Camera start error:', error);
            this.showError(`AI service error: ${error.message}`);
        }
    }

    async stopAICamera() {
        try {
            if (this.isAIActive) {
                await fetch(`${this.flaskUrl}/stop_camera`, { method: 'POST' });
                this.isAIActive = false;
            }
            
            this.aiCameraFeed.src = '';
            this.aiCameraFeed.style.display = 'none';
            this.stopStatusUpdates();
            
            console.log('AI camera stopped');
        } catch (error) {
            console.error('Error stopping AI camera:', error);
        }
    }

    startStatusUpdates() {
        this.stopStatusUpdates(); // Clear any existing interval
        
        this.statusUpdateInterval = setInterval(async () => {
            try {
                const response = await fetch(`${this.flaskUrl}/status`);
                
                if (response.ok) {
                    const data = await response.json();
                    this.updateStatus(data);
                }
            } catch (error) {
                console.error('Status update error:', error);
            }
        }, 1000); // Update every second
    }

    stopStatusUpdates() {
        if (this.statusUpdateInterval) {
            clearInterval(this.statusUpdateInterval);
            this.statusUpdateInterval = null;
        }
    }

    updateStatus(data) {
        // Update prediction
        if (data.prediction && data.prediction !== "Waiting...") {
            this.updateDetectedPhrase(`"${data.prediction}"`);
        } else {
            this.detectedPhrase.textContent = 'Waiting for AI detection...';
        }
        
        // Update confidence
        const confidence = data.confidence || 0;
        this.confidenceStatus.textContent = confidence.toFixed(3);
        
        // Update confidence color class
        this.confidenceStatus.className = confidence > 0.7 ? 'confidence-high' :
                                         confidence > 0.3 ? 'confidence-medium' : 'confidence-low';
        
        // Update buffer status
        const bufferText = data.buffer_ready ? 
            `Ready (${data.buffer_size}/16)` : 
            `Collecting (${data.buffer_size}/16)`;
        this.bufferStatus.textContent = bufferText;
    }

    async toggleHandDetection() {
        try {
            const response = await fetch(`${this.flaskUrl}/toggle_hands`, { method: 'POST' });
            const data = await response.json();
            
            const btn = document.getElementById('toggleHands');
            btn.textContent = data.show_hands ? 'Hide Hands' : 'Show Hands';
            btn.className = data.show_hands ? 'option-btn active' : 'option-btn';
            
            console.log('Hand detection toggled:', data.show_hands);
        } catch (error) {
            console.error('Error toggling hand detection:', error);
        }
    }

    async resetBuffer() {
        try {
            const response = await fetch(`${this.flaskUrl}/reset_detector`, { method: 'POST' });
            
            if (response.ok) {
                this.detectedPhrase.textContent = 'Buffer reset - collecting frames...';
                console.log('Buffer reset successfully');
            }
        } catch (error) {
            console.error('Error resetting buffer:', error);
        }
    }

    showLoading(show) {
        this.cameraLoading.style.display = show ? 'flex' : 'none';
        this.aiCameraFeed.style.display = show ? 'none' : 'block';
        this.aiStatus.style.display = show ? 'none' : 'flex';
    }

    showError(message = 'Unable to start AI camera. Please check if the AI service is running.') {
        this.cameraError.style.display = 'flex';
        this.cameraError.querySelector('p').textContent = message;
        this.cameraLoading.style.display = 'none';
        this.aiCameraFeed.style.display = 'none';
        this.aiStatus.style.display = 'none';
    }

    hideError() {
        this.cameraError.style.display = 'none';
    }

    handleResize() {
        // Handle any resize logic for the AI camera feed
        if (this.isExpanded) {
            // Adjust for fullscreen mode
        }
    }

    switchMode(mode) {
        this.currentMode = mode;
        
        // Update button states
        document.querySelectorAll('.mode-btn').forEach(btn => {
            btn.classList.remove('active');
        });
        
        if (mode === 'basic') {
            document.getElementById('basicDetect').classList.add('active');
        } else {
            document.getElementById('advancedDetect').classList.add('active');
        }
        
        console.log(`Switched to ${mode} detection mode`);
    }

    toggleExpanded() {
        this.isExpanded = !this.isExpanded;
        
        if (this.isExpanded) {
            this.cameraSection.classList.add('expanded');
            document.body.style.overflow = 'hidden';
            // Update expand button icon to show "contract" symbol
            document.getElementById('expandBtn').innerHTML = `
                <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                    <path d="M8 3v3a2 2 0 0 1-2 2H3m18 0h-3a2 2 0 0 1-2-2V3m0 18v-3a2 2 0 0 1 2-2h3M3 16h3a2 2 0 0 1 2 2v3"/>
                </svg>
            `;
        } else {
            this.cameraSection.classList.remove('expanded');
            document.body.style.overflow = '';
            // Revert to expand icon
            document.getElementById('expandBtn').innerHTML = `
                <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                    <path d="M8 3H5a2 2 0 0 0-2 2v3m18 0V5a2 2 0 0 0-2-2h-3m0 
                            18h3a2 2 0 0 0 2-2v-3M3 16v3a2 2 0 0 0 2 
                            2h3"/>
                </svg>
            `;
        }
    }

    updateDetectedPhrase(phrase) {
        // Animate the phrase update
        this.detectedPhrase.style.opacity = '0.5';
        this.detectedPhrase.style.transform = 'scale(0.95)';
        
        setTimeout(() => {
            this.detectedPhrase.textContent = phrase;
            this.detectedPhrase.style.opacity = '1';
            this.detectedPhrase.style.transform = 'scale(1)';
        }, 150);
    }

    playDetectedPhrase() {
        const playBtn = document.getElementById('playBtn');
        const playIcon = playBtn.querySelector('svg polygon');
        const playText = playBtn.querySelector('span');
        
        // Simulate audio playing
        playBtn.classList.add('playing');
        playText.textContent = 'Playing...';
        
        // Change icon to pause
        playIcon.setAttribute('points', '6,4 6,20 10,20 10,4 14,4 14,20 18,20 18,4');
        
        // Simulate playback duration (2 seconds)
        setTimeout(() => {
            this.resetPlayButton(playBtn, playIcon, playText);
            
            // Show modal after playback
            document.getElementById('responseModal').style.display = 'flex';
        }, 2000);
    }

    resetPlayButton(playBtn, playIcon, playText) {
        playBtn.classList.remove('playing');
        playText.textContent = 'Play audio';
        playIcon.setAttribute('points', '5,3 19,12 5,21');
    }

    toggleStar() {
        const starBtn = document.getElementById('starBtn');
        const isStarred = starBtn.classList.contains('active');
        
        if (isStarred) {
            starBtn.classList.remove('active');
            console.log('Removed from favorites');
        } else {
            starBtn.classList.add('active');
            console.log('Added to favorites');
            
            // Animation after the star is clicked
            starBtn.style.transform = 'scale(1.2)';
            setTimeout(() => {
                starBtn.style.transform = 'scale(1)';
            }, 200);
        }
    }

    // Cleanup method
    async destroy() {
        this.stopStatusUpdates();
        await this.stopAICamera();
        console.log('AI Sign Language Detection destroyed');
    }
}

// Initialize the AI sign language detection when the page loads
document.addEventListener('DOMContentLoaded', () => {
    const detector = new AISignLanguageDetection();
    
    // Handle page unload
    window.addEventListener('beforeunload', () => {
        detector.destroy();
    });
    
    // Handle page visibility changes
    document.addEventListener('visibilitychange', () => {
        if (document.hidden) {
            // Page is hidden, you might want to pause updates
            console.log('Page hidden - AI detection continues in background');
        } else {
            // Page is visible again
            console.log('Page visible - AI detection active');
        }
    });
});