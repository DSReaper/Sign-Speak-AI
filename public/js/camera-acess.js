class AISignLanguageDetection {
    constructor() {
        this.aiCameraFeed = document.getElementById('aiCameraFeed');
        this.cameraSection = document.getElementById('cameraSection');
        this.cameraLoading = document.getElementById('cameraLoading');
        this.cameraError = document.getElementById('cameraError');
        this.detectedPhrase = document.getElementById('detectedPhrase');
    // buffer and confidence UI removed — keep optional references guarded
    this.bufferStatus = document.getElementById('bufferStatus');
    this.confidenceStatus = document.getElementById('confidenceStatus');
        this.aiStatus = document.getElementById('aiStatus');
        
        this.isExpanded = false;
        this.isAIActive = false;
        this.currentMode = 'basic';
        // Performance tuning (can be toggled)
        this.performanceMode = 'balanced'; // 'quality' | 'balanced' | 'speed'
        this._perfSettings = {
            quality: { fps: 6, sendWidth: 640, jpegQuality: 0.8 },
            balanced: { fps: 8, sendWidth: 320, jpegQuality: 0.6 },
            speed: { fps: 10, sendWidth: 224, jpegQuality: 0.5 }
        };
    // WebSocket URL to Flask/AI server (served from Node.js proxy or directly)
    this.wsUrl = (location.protocol === 'https:' ? 'wss://' : 'ws://') + location.hostname + ':5001/ai/ws';
    this.flaskUrl = 'http://localhost:8001/ai'; // legacy HTTP endpoints still used for status controls
        this.statusUpdateInterval = null;
        this._shouldReconnect = true;
        this._reconnectAttempts = 0;
        
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

        // Performance mode selector
        const perfSel = document.getElementById('perfModeSelect');
        if (perfSel) {
            perfSel.addEventListener('change', (e) => {
                const val = e.target.value;
                if (['quality', 'balanced', 'speed'].includes(val)) {
                    this.performanceMode = val;
                    console.log('Performance mode set to', val);
                    // Restart capture interval to apply the new FPS / resolution / quality
                    // Use the internal helper so we don't rely on WebSocket lifecycle hooks.
                    if (this.captureInterval) {
                        clearInterval(this.captureInterval);
                        this.captureInterval = null;
                    }
                    // If ws is open, start the interval immediately. Otherwise, it will be
                    // started when the socket connects.
                    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
                        this._startCaptureInterval();
                    }

                    // Provide quick UI feedback for the selected mode
                    const perfLabel = document.getElementById('perfModeLabel');
                    if (perfLabel) perfLabel.textContent = val.charAt(0).toUpperCase() + val.slice(1);
                }
            });
        }
    }

    async startAICamera() {
        try {
            this.showLoading(true);
            this.hideError();
            
            console.log('Starting AI camera...');
            // Start camera capture from user's browser
            this._shouldReconnect = true;
            await this._startLocalCameraAndWebSocket();
            this.showLoading(false);
            this.isAIActive = true;
            this.startStatusUpdates();
            
        } catch (error) {
            console.error('AI Camera start error:', error);
            this.showError(`AI service error: ${error.message}`);
        }
    }

    async stopAICamera() {
        try {
            // Stop local capture and websocket
            if (this.captureStream) {
                this.captureStream.getTracks().forEach(t => t.stop());
                this.captureStream = null;
            }

            if (this.captureInterval) {
                clearInterval(this.captureInterval);
                this.captureInterval = null;
            }

            if (this.ws && this.ws.readyState === WebSocket.OPEN) {
                try { this.ws.close(); } catch (e) {}
            }

            // Prevent reconnect attempts when user stops camera
            this._shouldReconnect = false;

            this.isAIActive = false;
            this.aiCameraFeed.src = '';
            this.aiCameraFeed.style.display = 'none';
            this.stopStatusUpdates();

            console.log('AI camera stopped (local)');
        } catch (error) {
            console.error('Error stopping AI camera:', error);
        }
    }

    async _startLocalCameraAndWebSocket() {
        // Acquire user camera
        // This requests camera access from the browser. The stream returned is stored
        // in `this.captureStream` and later used as the source for a hidden <video>
        // element. We capture frames from that hidden element and send them to the
        // AI WebSocket server for low-latency processing.
        try {
            this.captureStream = await navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 480, facingMode: 'user' }, audio: false });
        } catch (err) {
            throw new Error('Unable to access camera: ' + err.message);
        }

    // Create hidden video element to draw frames
    // The hidden video (`this._hiddenVideo`) plays the live MediaStream. We draw
    // frames from that video onto an offscreen canvas (`this._canvas`) and then
    // downscale/encode those frames before sending them over the WebSocket.
    // Keeping the video hidden prevents visual duplication while preserving
    // full access to decoded video frames.
        if (!this._hiddenVideo) {
            this._hiddenVideo = document.createElement('video');
            this._hiddenVideo.autoplay = true;
            this._hiddenVideo.muted = true;
            this._hiddenVideo.playsInline = true;
            this._canvas = document.createElement('canvas');
            this._ctx = this._canvas.getContext('2d');
        }

        this._hiddenVideo.srcObject = this.captureStream;

    // Wait for video to be ready
    // We await the video playback to have actual frames available (onplaying).
    // There's a 3s timeout to avoid locking the UI if playback stalls on some
    // browsers. If the timeout triggers we continue anyway but frames may be
    // empty or black until playback actually starts.
        await new Promise((resolve, reject) => {
            this._hiddenVideo.onloadedmetadata = () => {
                this._canvas.width = this._hiddenVideo.videoWidth || 640;
                this._canvas.height = this._hiddenVideo.videoHeight || 480;
                // Start playback explicitly and wait for 'playing' to ensure frames are available
                const playPromise = this._hiddenVideo.play();
                if (playPromise && typeof playPromise.then === 'function') {
                    playPromise.then(() => {
                        // Wait for a real playing event
                        this._hiddenVideo.onplaying = () => {
                            console.log('Hidden video playing - frames available');
                            resolve();
                        };
                    }).catch((err) => {
                        console.warn('Hidden video play() failed:', err);
                        // still resolve to avoid blocking, but frames may be black
                        resolve();
                    });
                } else {
                    // Fallback: wait for onplaying
                    this._hiddenVideo.onplaying = () => {
                        console.log('Hidden video playing (no promise) - frames available');
                        resolve();
                    };
                }
            };
            // Timeout to avoid hanging
            setTimeout(() => {
                console.warn('Timed out waiting for hidden video to start - proceeding');
                resolve();
            }, 3000);
        });

        // Open WebSocket to AI server for low-latency frame processing
        // WebSocket lifecycle:
        //  - onopen: mark connection ready and start the capture interval
        //  - onmessage: receive processed JPEG bytes from server and display them
        //  - onerror/onclose: attempt reconnects using exponential backoff when
        //    `this._shouldReconnect` is true. When reconnecting we preserve the
        //    local camera capture so the user still sees the last frames.
        // If there's an existing ws, close it first
        if (this.ws && (this.ws.readyState === WebSocket.OPEN || this.ws.readyState === WebSocket.CONNECTING)) {
            try { this.ws.close(); } catch (e) { console.warn('Error closing previous ws', e); }
        }
        this.ws = new WebSocket(this.wsUrl);

        // Helper to update the small colored dot and connection text inside the ai-status element
        // state: 'connected' | 'connecting' | 'disconnected'
        // text: optional display text
        this.setConnectionState = (state, text) => {
            try {
                const dot = document.getElementById('connectionStatusDot');
                const txt = document.getElementById('connectionStatusText');
                if (dot) {
                    dot.classList.remove('connected', 'connecting', 'disconnected');
                    if (state === 'connected') dot.classList.add('connected');
                    else if (state === 'connecting') dot.classList.add('connecting');
                    else dot.classList.add('disconnected');
                }
                if (txt) txt.textContent = text || (state === 'connected' ? 'Connected' : state === 'connecting' ? 'Connecting...' : 'Disconnected');
                if (this.aiStatus) this.aiStatus.style.display = 'flex';
            } catch (err) { /* ignore UI update errors */ }
        };

    // initial state while the WS connection is being established
    // Keep dot state only; we do not render any textual overlays above the feed
    this.setConnectionState('connecting');

        // Basic backpressure: only send a new frame when we don't have a pending request
        this._pending = false;

        this.ws.binaryType = 'arraybuffer';

        this.ws.onopen = () => {
            // WebSocket is ready for binary frame transfer.
            // Reset reconnect attempts and update short status. We don't hide the
            // camera UI here because the camera is local and should remain visible
            // even if the AI server was temporarily unreachable.
            console.log('WebSocket connected to AI detection service');
            this._reconnectAttempts = 0;
                try {
                    this.hideError();
                    // update connection dot only
                    this.setConnectionState('connected');
                } catch (e) { /* ignore UI update errors */ }

            // Start sending frames at the configured FPS/resolution.
            this._startCaptureInterval();
        };

        // When the socket closes, attempt reconnection unless user stopped
    this.ws.onclose = (evt) => {
            // Log close code and reason when available for debugging
            try {
                console.warn('WebSocket closed', {
                    code: evt.code,
                    reason: evt.reason,
                    wasClean: evt.wasClean
                });
            } catch (err) {
                console.warn('WebSocket closed (no evt details available)');
            }
            this._pending = false;

            // When the socket closes, don't immediately hide the camera. If
            // `_shouldReconnect` is true we keep the local video running and try
            // to re-establish the WebSocket after an exponential backoff. This
            // avoids blinking the UI while the server restarts.
            if (this._shouldReconnect) {
                const backoff = Math.min(30, Math.pow(2, this._reconnectAttempts));
                console.log(`WebSocket closed unexpectedly — reconnecting in ${backoff}s (attempt ${this._reconnectAttempts + 1})`);
                try {
                    // show connecting state with countdown
                    this.setConnectionState('connecting', `Disconnected — reconnecting in ${backoff}s`);
                } catch (e) { /* ignored */ }

                setTimeout(() => {
                    this._reconnectAttempts += 1;
                    // Re-create the WS connection but reuse the same local camera
                    this._startLocalCameraAndWebSocket().catch(err => console.warn('Reconnect failed', err));
                }, backoff * 1000);
            } else {
                // If reconnecting is disabled (user stopped the camera), show
                // a full error message to the user so they can take action.
                if (!this.isAIActive) {
                    // mark disconnected
                    this.setConnectionState('disconnected', 'Disconnected');
                    this.showError('AI WebSocket connection closed by server. Ensure the Flask AI server is running and accepting ws connections on port 5001.');
                }
            }
        };

        this.ws.onmessage = (evt) => {
            // Server will send back a processed JPEG image (binary). We expect
            // binary frames (Blob or ArrayBuffer). Text messages are ignored or
            // logged.
            const data = evt.data;

            if (!(data instanceof ArrayBuffer) && !(data instanceof Blob)) {
                // Unexpected message type — log and ignore.
                console.warn('WS: received unexpected non-binary message', data);
                this._pending = false;
                return;
            }

            // Convert received bytes into a blob URL and set it as the <img>
            // `src`. Setting `aiCameraFeed.src` to a blob URL is quick and lets
            // the browser decode and display the JPEG. We also draw the image to
            // a hidden fallback canvas for pixel inspection/debugging.
            const blob = data instanceof Blob ? data : new Blob([data], { type: 'image/jpeg' });
            console.log('WS: received frame blob, size=', blob.size);

            // Previously we flashed a green border here on each incoming frame;
            // that caused a visible green outline. Remove inline border updates
            // to avoid the green border flash and let CSS handle styling.
            // If a frame-arrival indicator is desired, add/remove a CSS class
            // instead of setting inline styles.

            if (!this._fallbackCanvas) {
                this._fallbackCanvas = document.createElement('canvas');
                this._fallbackCanvasCtx = this._fallbackCanvas.getContext('2d');
                this._fallbackCanvas.style.display = 'none';
                document.body.appendChild(this._fallbackCanvas);
            }

            // Use object URL because it's fast and avoids copying large binary
            // arrays into base64 strings. We revoke the URL after the image is
            // loaded to free memory.
            const url = URL.createObjectURL(blob);
            this.aiCameraFeed.src = url;
            this.aiCameraFeed.style.display = 'block';

            // Optionally decode and draw to fallback canvas (useful for
            // diagnostics or pixel-level checks). This does not replace the
            // visible <img> which the user sees.
            const img = new Image();
            img.onload = () => {
                try {
                    this._fallbackCanvas.width = img.width;
                    this._fallbackCanvas.height = img.height;
                    this._fallbackCanvasCtx.drawImage(img, 0, 0);
                } catch (err) {
                    console.warn('Fallback canvas draw error', err);
                }
            };
            img.onerror = (err) => { console.warn('Image decode error on fallback', err); };
            img.src = url;

            // When the <img> has finished decoding and painting, revoke the URL
            // and clear the pending flag so the next frame can be sent.
            this.aiCameraFeed.onload = () => {
                try { URL.revokeObjectURL(url); } catch (err) { /* ignore */ }
                this._pending = false;
            };
        };

        this.ws.onerror = (e) => {
            console.error('WebSocket error', e, 'readyState=', this.ws ? this.ws.readyState : 'no-ws');
            // For transient errors while auto-reconnect is enabled, avoid hiding the camera
            if (this._shouldReconnect) {
                console.warn('Transient WebSocket error — will attempt reconnect');
                try {
                    this.setConnectionState('connecting', 'AI server connection error — reconnecting...');
                } catch (uiErr) { /* ignore */ }
            } else {
                // If reconnect is disabled (user stopped the camera), show a full error UI
                this.showError('WebSocket error connecting to AI server. See console for details.');
            }
        };

        // Helper to update the small colored dot and connection text inside the ai-status element
        // state: 'connected' | 'connecting' | 'disconnected'
        // text: optional display text
        this.setConnectionState = (state, text) => {
            try {
                const dot = document.getElementById('connectionStatusDot');
                const txt = document.getElementById('connectionStatusText');
                if (dot) {
                    dot.classList.remove('connected', 'connecting', 'disconnected');
                    if (state === 'connected') dot.classList.add('connected');
                    else if (state === 'connecting') dot.classList.add('connecting');
                    else dot.classList.add('disconnected');
                }
                if (txt) txt.textContent = text || (state === 'connected' ? 'Connected' : state === 'connecting' ? 'Connecting...' : 'Disconnected');
                if (this.aiStatus) this.aiStatus.style.display = 'flex';
            } catch (err) { /* ignore UI update errors */ }
        };

        // Note: onclose handled above to manage reconnect
    }

    _startCaptureInterval() {
        // Clear existing interval if any
        if (this.captureInterval) {
            clearInterval(this.captureInterval);
            this.captureInterval = null;
        }

        const perf = this._perfSettings[this.performanceMode] || this._perfSettings.balanced;
        const targetFps = perf.fps;
        const intervalMs = 1000 / targetFps;

        // Log the applied perf settings for debugging
        console.log('Starting capture interval with settings:', {
            performanceMode: this.performanceMode,
            fps: perf.fps,
            sendWidth: perf.sendWidth,
            jpegQuality: perf.jpegQuality
        });

        this.captureInterval = setInterval(async () => {
            // Backpressure: only send if there is no pending server response and
            // the socket is open. `_pending` prevents flooding the server while
            // a previous frame is still being processed.
            if (this._pending || this.ws.readyState !== WebSocket.OPEN) return;

            // Ensure the hidden video has data before drawing. readyState >= 2
            // means the element has some decoded frames available.
            if (!this._hiddenVideo || this._hiddenVideo.readyState < 2) {
                return;
            }

            // Draw current decoded frame to the offscreen canvas. This gives us
            // an ImageBitmap-like pixel source we can resize and re-encode.
            this._ctx.drawImage(this._hiddenVideo, 0, 0, this._canvas.width, this._canvas.height);

            // Resize to configured resolution for transfer (bandwidth tradeoff).
            const perfSet = this._perfSettings[this.performanceMode] || this._perfSettings.balanced;
            const sendWidth = perfSet.sendWidth;
            const sendHeight = Math.round((this._canvas.height / this._canvas.width) * sendWidth);

            // Use a temporary canvas to perform resizing. This keeps the main
            // canvas at camera resolution while sending a smaller JPEG over the
            // network to reduce bandwidth and processing time on the server.
            const off = document.createElement('canvas');
            off.width = sendWidth;
            off.height = sendHeight;
            off.getContext('2d').drawImage(this._canvas, 0, 0, sendWidth, sendHeight);

            // Encode as JPEG with configured quality. toBlob is async and
            // returns a Blob which we send directly over the WebSocket. We set
            // `_pending` true before send and clear it when the server responds
            // (in ws.onmessage) so we don't overlap frames.
            const jpegQuality = perfSet.jpegQuality || 0.6;
            off.toBlob((blob) => {
                if (!blob) return;
                this._pending = true;
                try {
                    this.ws.send(blob);
                } catch (err) {
                    console.warn('Failed to send blob over WS', err);
                    // If send fails immediately, clear pending so future frames
                    // can attempt to send again.
                    this._pending = false;
                }
            }, 'image/jpeg', jpegQuality);
        }, intervalMs);
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
        // Buffer and confidence UI were removed; do not attempt to update DOM
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