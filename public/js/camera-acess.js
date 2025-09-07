    class SignLanguageDetection {
            constructor() {
                this.video = document.getElementById('cameraFeed');
                this.canvas = document.getElementById('overlay');
                this.ctx = this.canvas.getContext('2d');
                this.cameraSection = document.getElementById('cameraSection');
                this.cameraLoading = document.getElementById('cameraLoading');
                this.cameraError = document.getElementById('cameraError');
                this.detectedPhrase = document.getElementById('detectedPhrase');
                
                this.stream = null;
                this.isExpanded = false;
                this.isDetecting = false;
                this.currentMode = 'basic';
                
                this.initializeEventListeners();
                this.startCamera();
            }

            initializeEventListeners() {
                // Mode switching
                document.getElementById('basicDetect').addEventListener('click', () => {
                    this.switchMode('basic');
                });
                
                document.getElementById('advancedDetect').addEventListener('click', () => {
                    this.switchMode('advanced');
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
                    this.startCamera();
                });
                
                // Handle window resize
                window.addEventListener('resize', () => {
                    this.resizeCanvas();
                });
                
                // Handle escape key for fullscreen
                document.addEventListener('keydown', (e) => {
                    if (e.key === 'Escape' && this.isExpanded) {
                        this.toggleExpanded();
                    }
                });

                document.getElementById('closeModal').addEventListener('click', () => {
                    const modal = document.getElementById('responseModal');
                    const textarea = document.getElementById('userResponse');
                    
                    // Hide modal
                    modal.style.display = 'none';
                    
                    // Clear textarea contents
                    textarea.value = '';
                });

            }

            async startCamera() {
                try {
                    this.showLoading(true);
                    this.hideError();
                    
                    // Stop existing stream if any
                    if (this.stream) {
                        this.stream.getTracks().forEach(track => track.stop());
                    }
                    
                    // Request camera access
                    this.stream = await navigator.mediaDevices.getUserMedia({
                        video: { 
                            width: { ideal: 1280 },
                            height: { ideal: 720 },
                            facingMode: 'user'
                        },
                        audio: false
                    });
                    
                    this.video.srcObject = this.stream;
                    
                    // Wait for video to load
                    this.video.addEventListener('loadedmetadata', () => {
                        this.showLoading(false);
                        this.resizeCanvas();
                        this.startDetection();
                    });
                    
                    this.video.addEventListener('error', (e) => {
                        console.error('Video error:', e);
                        this.showError();
                    });
                    
                } catch (error) {
                    console.error('Camera access error:', error);
                    this.showError();
                }
            }

            showLoading(show) {
                this.cameraLoading.style.display = show ? 'flex' : 'none';
                this.video.style.display = show ? 'none' : 'block';
            }

            showError() {
                this.cameraError.style.display = 'flex';
                this.cameraLoading.style.display = 'none';
                this.video.style.display = 'none';
            }

            hideError() {
                this.cameraError.style.display = 'none';
            }

            resizeCanvas() {
                if (this.video.videoWidth && this.video.videoHeight) {
                    this.canvas.width = this.video.videoWidth;
                    this.canvas.height = this.video.videoHeight;
                    this.canvas.style.width = '100%';
                    this.canvas.style.height = '100%';
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
                
                // Restart detection with new mode
                this.startDetection();
                
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
                
                // Resize canvas after transition
                setTimeout(() => {
                    this.resizeCanvas();
                }, 400);
            }

            startDetection() {
                if (this.isDetecting) return;
                
                this.isDetecting = true;
                console.log(`Starting ${this.currentMode} sign detection...`);
                
                // Simulate detection process
                this.detectSigns();
            }

            detectSigns() {

                const phrases = [
                    '"Hi how are you doing today?"',
                ];
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
                    
                    // A animation after the star is clicked
                    starBtn.style.transform = 'scale(1.2)';
                    setTimeout(() => {
                        starBtn.style.transform = 'scale(1)';
                    }, 200);
                }
            }

            // Cleanup method
            destroy() {
                this.isDetecting = false;
                
                if (this.stream) {
                    this.stream.getTracks().forEach(track => track.stop());
                }
            }
        }

        // Initialize the sign language detection when the page loads
        document.addEventListener('DOMContentLoaded', () => {
            const detector = new SignLanguageDetection();
            
            // Handle page unload
            window.addEventListener('beforeunload', () => {
                detector.destroy();
            });
        });