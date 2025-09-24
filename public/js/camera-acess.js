// Fallback hand connection list
const HAND_CONNECTIONS_FALLBACK = [
    [0,1],[1,2],[2,3],[3,4],
    [0,5],[5,6],[6,7],[7,8],
    [5,9],[9,10],[10,11],[11,12],
    [9,13],[13,14],[14,15],[15,16],
    [13,17],[17,18],[18,19],[19,20],
    [0,17]
];

class AISignLanguageDetection {
    constructor() {
        // visible feed element
        this.aiCameraFeed = document.getElementById('aiCameraFeed');
        this.cameraSection = document.getElementById('cameraSection');
        this.cameraLoading = document.getElementById('cameraLoading');
        this.cameraError = document.getElementById('cameraError');
        this.detectedPhrase = document.getElementById('detectedPhrase');
        // overlay canvas for landmarks
        this.overlayCanvas = document.getElementById('overlayCanvas');
        this.overlayCtx = this.overlayCanvas ? this.overlayCanvas.getContext('2d') : null;
        // Enable local overlay by default
        this._localHandsEnabled = true;
        if (this.overlayCanvas) {
            this.overlayCanvas.style.display = 'block';
            this.overlayCanvas.style.zIndex = '5';
        }
        // optional UI elements (may be absent)
    this.bufferStatus = document.getElementById('bufferStatus');
    this.confidenceStatus = document.getElementById('confidenceStatus');
        this.aiStatus = document.getElementById('aiStatus');
        
        this.isExpanded = false;
        this.isAIActive = false;
        this.currentMode = 'basic';
        // performance presets
        this.performanceMode = 'balanced'; // 'quality' | 'balanced' | 'speed'
        this._perfSettings = {
            quality: { fps: 6, sendWidth: 640, jpegQuality: 0.8 },
            balanced: { fps: 8, sendWidth: 320, jpegQuality: 0.6 },
            speed: { fps: 10, sendWidth: 224, jpegQuality: 0.5 }
        };
        // reusable send canvas
        this._sendCanvas = null;
        this._sendCtx = null;
        // backpressure helpers
        this._lastSent = 0;
        this._bufferedThreshold = 1e6; // 1 MB queued => drop frames
        this._pendingTimeout = null; // used to clear _pending if server stalls
        // WebSocket and HTTP endpoints
        this.wsUrl = (location.protocol === 'https:' ? 'wss://' : 'ws://') + location.hostname + ':5001/ai/ws';
        this.flaskUrl = 'http://localhost:8001/ai';
        this.statusUpdateInterval = null;
        this._shouldReconnect = true;
        this._reconnectAttempts = 0;
        
        this.initializeEventListeners();
        this.startAICamera();
    }

    initializeEventListeners() {
        // mode switching
        document.getElementById('basicDetect').addEventListener('click', () => {
            this.switchMode('basic');
        });
        
        document.getElementById('advancedDetect').addEventListener('click', () => {
            this.switchMode('advanced');
        });
        
        // AI controls
        document.getElementById('toggleHands').addEventListener('click', () => {
            // toggle local overlay and inform server
            this._localHandsEnabled = !this._localHandsEnabled;
            if (this.overlayCanvas) {
                this.overlayCanvas.style.display = this._localHandsEnabled ? 'block' : 'none';
            }
            // Also inform server (preserve existing behaviour)
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
        
        // window resize
        window.addEventListener('resize', () => {
            this.handleResize();
        });
        
        // escape key to contract fullscreen
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

        // performance selector
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
            // If the visible feed is a video element, stop and clear its srcObject
            try {
                if (this.aiCameraFeed && this.aiCameraFeed.tagName && this.aiCameraFeed.tagName.toLowerCase() === 'video') {
                    try { this.aiCameraFeed.pause(); } catch (e) { /* ignore */ }
                    try { this.aiCameraFeed.srcObject = null; } catch (e) { /* ignore */ }
                } else if (this.aiCameraFeed) {
                    // legacy img element
                    try { this.aiCameraFeed.src = ''; } catch (e) { /* ignore */ }
                }
            } catch (err) { /* ignore */ }
            if (this.aiCameraFeed) this.aiCameraFeed.style.display = 'none';
            this.stopStatusUpdates();

            console.log('AI camera stopped (local)');
        } catch (error) {
            console.error('Error stopping AI camera:', error);
        }
    }

    async _startLocalCameraAndWebSocket() {
        // Acquire user camera and prepare local video/canvas
        try {
            // Choose capture constraints based on performance preference. We still
            // request a reasonably-sized camera feed (so browser decoding is good)
            // but we will downscale before sending. Lowering the capture
            // resolution can reduce CPU on some devices.
            const capConstraints = (this.performanceMode === 'quality') ? { width: 1280, height: 720 } : { width: 640, height: 480 };
            this.captureStream = await navigator.mediaDevices.getUserMedia({ video: Object.assign({ facingMode: 'user' }, capConstraints), audio: false });
        } catch (err) {
            throw new Error('Unable to access camera: ' + err.message);
        }

        // Create hidden video and offscreen canvas for encoding
        if (!this._hiddenVideo) {
            this._hiddenVideo = document.createElement('video');
            this._hiddenVideo.autoplay = true;
            this._hiddenVideo.muted = true;
            this._hiddenVideo.playsInline = true;
            this._canvas = document.createElement('canvas');
            this._ctx = this._canvas.getContext('2d');
        }

        // Bind MediaStream to visible video
        try {
            // If aiCameraFeed is a <video>, set its srcObject so it displays the camera directly
            if (this.aiCameraFeed && this.aiCameraFeed.tagName && this.aiCameraFeed.tagName.toLowerCase() === 'video') {
                this.aiCameraFeed.srcObject = this.captureStream;
                // keep a hidden video for encoding/sending if needed
                this._hiddenVideo.srcObject = this.captureStream;
            } else {
                // fallback: keep original behaviour
                this._hiddenVideo.srcObject = this.captureStream;
            }
        } catch (err) {
            this._hiddenVideo.srcObject = this.captureStream;
        }

        // Initialize local MediaPipe Hands for overlay
        this._localHandsEnabled = true;
        this._initLocalHands();

        // Wait for hidden video to be playing (3s timeout)
        await new Promise((resolve, reject) => {
            this._hiddenVideo.onloadedmetadata = () => {
                this._canvas.width = this._hiddenVideo.videoWidth || 640;
                this._canvas.height = this._hiddenVideo.videoHeight || 480;
                // Create a reusable send canvas sized to the maximum expected
                // send width; we will resize into this canvas to avoid
                // allocating many temporary elements per frame.
                if (!this._sendCanvas) {
                    this._sendCanvas = document.createElement('canvas');
                    this._sendCtx = this._sendCanvas.getContext('2d');
                }
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

        // Open WebSocket for frame processing
        if (this.ws && (this.ws.readyState === WebSocket.OPEN || this.ws.readyState === WebSocket.CONNECTING)) {
            try { this.ws.close(); } catch (e) { console.warn('Error closing previous ws', e); }
        }
        this.ws = new WebSocket(this.wsUrl);

        // Helper to update connection UI
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

        // initial WS state
        this.setConnectionState('connecting');

        // backpressure: only send when not pending
        this._pending = false;

        this.ws.binaryType = 'arraybuffer';

        this.ws.onopen = () => {
            console.log('WebSocket connected to AI detection service');
            this._reconnectAttempts = 0;
            try { this.hideError(); this.setConnectionState('connected'); } catch (e) { }
            this._startCaptureInterval();
        };

        // onclose: reconnect logic handled here
        this.ws.onclose = (evt) => {
            try { console.warn('WebSocket closed', { code: evt.code, reason: evt.reason, wasClean: evt.wasClean }); } catch (err) { console.warn('WebSocket closed'); }
            this._pending = false;
            if (this._shouldReconnect) {
                const backoff = Math.min(30, Math.pow(2, this._reconnectAttempts));
                console.log(`WebSocket closed — reconnecting in ${backoff}s (attempt ${this._reconnectAttempts + 1})`);
                try { this.setConnectionState('connecting', `Disconnected — reconnecting in ${backoff}s`); } catch (e) { }
                setTimeout(() => {
                    this._reconnectAttempts += 1;
                    this._startLocalCameraAndWebSocket().catch(err => console.warn('Reconnect failed', err));
                }, backoff * 1000);
            } else {
                if (!this.isAIActive) {
                    this.setConnectionState('disconnected', 'Disconnected');
                    this.showError('AI WebSocket connection closed by server. Ensure the Flask AI server is running and accepting ws connections on port 5001.');
                }
            }
        };

        this.ws.onmessage = (evt) => {
            // server sends JPEG frames (Blob or ArrayBuffer)
            const data = evt.data;

            if (!(data instanceof ArrayBuffer) && !(data instanceof Blob)) {
                // Unexpected message type — log and ignore.
                console.warn('WS: received unexpected non-binary message', data);
                this._pending = false;
                return;
            }

            // convert bytes into a Blob and draw to fallback canvas for diagnostics
            const blob = data instanceof Blob ? data : new Blob([data], { type: 'image/jpeg' });
            console.log('WS: received frame blob, size=', blob.size);

            // avoid inline visual flashes; use CSS classes if needed

            if (!this._fallbackCanvas) {
                this._fallbackCanvas = document.createElement('canvas');
                this._fallbackCanvasCtx = this._fallbackCanvas.getContext('2d');
                this._fallbackCanvas.style.display = 'none';
                document.body.appendChild(this._fallbackCanvas);
            }

            // create object URL and draw to fallback canvas
            const url = URL.createObjectURL(blob);
            if (!this._fallbackCanvas) {
                this._fallbackCanvas = document.createElement('canvas');
                this._fallbackCanvasCtx = this._fallbackCanvas.getContext('2d');
                this._fallbackCanvas.style.display = 'none';
                document.body.appendChild(this._fallbackCanvas);
            }

            // decode image for fallback canvas
            const img = new Image();
            img.onload = () => {
                try {
                    this._fallbackCanvas.width = img.width;
                    this._fallbackCanvas.height = img.height;
                    this._fallbackCanvasCtx.drawImage(img, 0, 0);
                } catch (err) {
                    console.warn('Fallback canvas draw error', err);
                }
                try { URL.revokeObjectURL(url); } catch (err) { /* ignore */ }
                this._pending = false;
            };
            img.onerror = (err) => {
                console.warn('Image decode error on fallback', err);
                this._pending = false;
            };
            img.src = url;
        };

        this.ws.onerror = (e) => {
            console.error('WebSocket error', e, 'readyState=', this.ws ? this.ws.readyState : 'no-ws');
            if (this._shouldReconnect) {
                console.warn('Transient WebSocket error — will attempt reconnect');
                try { this.setConnectionState('connecting', 'AI server connection error — reconnecting...'); } catch (uiErr) { }
            } else {
                this.showError('WebSocket error connecting to AI server. See console for details.');
            }
        };

        // Note: onclose handled above
    }

    // Initialize MediaPipe Hands and start overlay
    _initLocalHands() {
        if (typeof window.Hands === 'undefined') {
            console.warn('MediaPipe Hands not available - skipping local overlay');
            return;
        }

        if (this._hands) return;

        console.log('Initializing local MediaPipe Hands...');
        this._hands = new Hands({ locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${file}` });
        this._hands.setOptions({
            maxNumHands: 2,
            modelComplexity: 0,
            minDetectionConfidence: 0.6,
            minTrackingConfidence: 0.5
        });

        this._localHandsFrames = 0;
        this._localHandsLastTime = performance.now();
        this._localHandsLastFps = 0;

        this._hands.onResults((results) => {
            this._drawHands(results);
            this._localHandsFrames += 1;
            const now = performance.now();
            if (now - this._localHandsLastTime >= 500) {
                this._localHandsLastFps = Math.round((this._localHandsFrames * 1000) / (now - this._localHandsLastTime));
                this._localHandsFrames = 0;
                this._localHandsLastTime = now;
            }
            if (Math.random() < 0.01) console.log('Local hands onResults - hands=', results.multiHandLandmarks ? results.multiHandLandmarks.length : 0, 'fps=', this._localHandsLastFps);
        });

        this._localHandsSkip = 0;
        const processFrame = async () => {
            try {
                if (this._hiddenVideo && this._hiddenVideo.readyState >= 2 && this._localHandsEnabled) {
                    this._localHandsSkip = (this._localHandsSkip + 1) & 1;
                    if (this._localHandsSkip === 0) {
                        await this._hands.send({ image: this._hiddenVideo });
                    }
                } else if (this.overlayCanvas && !this._localHandsEnabled) {
                    this.overlayCtx && this.overlayCtx.clearRect(0, 0, this.overlayCanvas.width, this.overlayCanvas.height);
                }
            } catch (err) { }
            this._localHandsRaf = requestAnimationFrame(processFrame);
        };

        this._localHandsRaf = requestAnimationFrame(processFrame);
    }

    _drawHands(results) {
        if (!this.overlayCanvas || !this.overlayCtx) return;

    // Match overlay canvas size to visible aiCameraFeed
        const imgEl = this.aiCameraFeed;
        const rect = imgEl.getBoundingClientRect();

    // fallback to hidden video if not visible
        const displayedWidth = rect.width || (this._hiddenVideo ? this._hiddenVideo.videoWidth : 640);
        const displayedHeight = rect.height || (this._hiddenVideo ? this._hiddenVideo.videoHeight : 480);

    // Use video's intrinsic pixel size to map landmarks
        const videoW = (this._hiddenVideo && this._hiddenVideo.videoWidth) ? this._hiddenVideo.videoWidth : (this.aiCameraFeed.videoWidth || displayedWidth);
        const videoH = (this._hiddenVideo && this._hiddenVideo.videoHeight) ? this._hiddenVideo.videoHeight : (this.aiCameraFeed.videoHeight || displayedHeight);

    // Compute scale and offsets for object-fit: cover
        const scaleX = displayedWidth / videoW;
        const scaleY = displayedHeight / videoH;
        // For object-fit: cover use the larger scale to fill and crop
        const scale = Math.max(scaleX, scaleY);

        const scaledVideoWidth = videoW * scale;
        const scaledVideoHeight = videoH * scale;
    // Offsets due to cropping
        const offsetX = (scaledVideoWidth - displayedWidth) / 2;
        const offsetY = (scaledVideoHeight - displayedHeight) / 2;

        let width = displayedWidth;
        let height = displayedHeight;

        // Detect if the displayed video is mirrored
        let isMirrored = false;
        try {
            const st = window.getComputedStyle(imgEl);
            const t = st && st.transform ? st.transform : '';
            if (t && t !== 'none') {
                // matrix(-1, 0, 0, 1, 0, 0) indicates horizontal flip
                if (t.indexOf('-1') !== -1) isMirrored = true;
            }
        } catch (e) {
            isMirrored = false;
        }

        // Resize overlay canvas if needed
        const dpr = window.devicePixelRatio || 1;
        if (this.overlayCanvas.width !== Math.round(width * dpr) || this.overlayCanvas.height !== Math.round(height * dpr)) {
            this.overlayCanvas.width = Math.round(width * dpr);
            this.overlayCanvas.height = Math.round(height * dpr);
            this.overlayCanvas.style.width = `${width}px`;
            this.overlayCanvas.style.height = `${height}px`;
            this.overlayCtx.setTransform(dpr, 0, 0, dpr, 0, 0);
        }

    // clear canvas
        this.overlayCtx.clearRect(0, 0, width, height);

        if (!results.multiHandLandmarks || results.multiHandLandmarks.length === 0) return;

    // Draw each hand's landmarks
        for (let i = 0; i < results.multiHandLandmarks.length; i++) {
            const landmarks = results.multiHandLandmarks[i];
            // Use drawing utilities if available
            if (window.drawConnectors && window.drawLandmarks) {
          
                const connections = (typeof window.HAND_CONNECTIONS !== 'undefined') ? window.HAND_CONNECTIONS : HAND_CONNECTIONS_FALLBACK;
                try {
  
                    const pixelLandmarks = landmarks.map((lm) => {
                        const xVideoPx = lm.x * videoW * scale; // scaled video pixels
                        const yVideoPx = lm.y * videoH * scale;
                        let x = xVideoPx - offsetX;
                        let y = yVideoPx - offsetY;
                        // Clamp to visible area
                        x = Math.max(0, Math.min(width, x));
                        y = Math.max(0, Math.min(height, y));
                        if (isMirrored) x = width - x;
                        // Preserve z/visibility if present
                        const out = { x, y };
                        if (typeof lm.z !== 'undefined') out.z = lm.z;
                        if (typeof lm.visibility !== 'undefined') out.visibility = lm.visibility;
                        return out;
                    });

                    this.overlayCtx.strokeStyle = 'rgba(0,255,0,0.9)';
                    this.overlayCtx.lineWidth = 2;
                    // Draw the connections
                    for (const c of connections) {
                        const a = pixelLandmarks[c[0]];
                        const b = pixelLandmarks[c[1]];
                        if (!a || !b) continue;
                        this.overlayCtx.beginPath();
                        this.overlayCtx.moveTo(a.x, a.y);
                        this.overlayCtx.lineTo(b.x, b.y);
                        this.overlayCtx.stroke();
                    }
                    // Draw landmarks as filled circles
                    this.overlayCtx.fillStyle = '#FF0000';
                    for (const p of pixelLandmarks) {
                        this.overlayCtx.beginPath();
                        this.overlayCtx.arc(p.x, p.y, 3, 0, Math.PI * 2);
                        this.overlayCtx.fill();
                    }
                } catch (err) {
                    // fallback to simple dots if drawing util fails
                    this.overlayCtx.fillStyle = 'red';
                    for (const lm of landmarks) {
                        const x = lm.x * width;
                        const y = lm.y * height;
                        this.overlayCtx.beginPath();
                        this.overlayCtx.arc(x, y, 3, 0, Math.PI * 2);
                        this.overlayCtx.fill();
                    }
                }
            } else {
                // Fallback simple drawing
                this.overlayCtx.fillStyle = 'red';
                for (const lm of landmarks) {
                    // Map normalized landmark (0..1) to CSS pixel coordinates taking
                    // into account intrinsic video size, scale, and cropping offset.
                    const xVideoPx = lm.x * videoW * scale; // scaled video pixels
                    const yVideoPx = lm.y * videoH * scale;
                    let x = xVideoPx - offsetX;
                    let y = yVideoPx - offsetY;
                    // Clamp to visible area
                    x = Math.max(0, Math.min(width, x));
                    y = Math.max(0, Math.min(height, y));
                    if (isMirrored) {
                        x = width - x;
                    }
                    this.overlayCtx.beginPath();
                    this.overlayCtx.arc(x, y, 3, 0, Math.PI * 2);
                    this.overlayCtx.fill();
                }
            }
            
            // Compute bounding box for this hand
            try {
                let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
                for (const lm of landmarks) {
                    // use the same mapping as in fallback drawing for accurate box
                    const xVideoPx = lm.x * videoW * scale;
                    const yVideoPx = lm.y * videoH * scale;
                    let x = xVideoPx - offsetX;
                    let y = yVideoPx - offsetY;
                    x = Math.max(0, Math.min(width, x));
                    y = Math.max(0, Math.min(height, y));
                    if (isMirrored) x = width - x;
                    if (x < minX) minX = x;
                    if (y < minY) minY = y;
                    if (x > maxX) maxX = x;
                    if (y > maxY) maxY = y;
                }
                if (minX !== Infinity) {
                    // Add small padding
                    const pad = Math.max(6, Math.min(24, Math.round((maxX - minX) * 0.08)));
                    minX = Math.max(0, minX - pad);
                    minY = Math.max(0, minY - pad);
                    maxX = Math.min(width, maxX + pad);
                    maxY = Math.min(height, maxY + pad);

                    // Draw box
                    this.overlayCtx.strokeStyle = 'rgba(0,255,0,0.9)';
                    this.overlayCtx.lineWidth = 2;
                    this.overlayCtx.strokeRect(minX + 0.5, minY + 0.5, (maxX - minX), (maxY - minY));
                }
            } catch (e) {
                // ignore bounding box errors
            }
        }

        // Draw debug info (fps and hand count)
        try {
            const handsCount = results.multiHandLandmarks ? results.multiHandLandmarks.length : 0;
            const fps = this._localHandsLastFps || 0;
            this.overlayCtx.font = '14px Arial';
            this.overlayCtx.fillStyle = 'rgba(0,0,0,0.6)';
            this.overlayCtx.fillRect(6, 6, 120, 26);
            this.overlayCtx.fillStyle = '#fff';
            this.overlayCtx.fillText(`hands: ${handsCount}  fps: ${fps}`, 12, 24);
        } catch (e) {  }
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

            // Reuse the send canvas to avoid per-frame allocation
            if (this._sendCanvas.width !== sendWidth || this._sendCanvas.height !== sendHeight) {
                this._sendCanvas.width = sendWidth;
                this._sendCanvas.height = sendHeight;
            }
            this._sendCtx.drawImage(this._canvas, 0, 0, sendWidth, sendHeight);

            // Encode as JPEG with configured quality. Avoid sending if the WS
            // backend is backed up (bufferedAmount high). This helps prevent
            // increasing latency by piling up queued bytes in the browser.
            const jpegQuality = perfSet.jpegQuality || 0.6;
            // If the socket's send buffer is large, skip this frame
            if (this.ws.bufferedAmount && this.ws.bufferedAmount > this._bufferedThreshold) {
                // drop frame to avoid growing send queue
                return;
            }

            // Mark pending and add a timeout to clear it in case the server
            // stalls and never replies (prevents permanent stuck state).
            this._pending = true;
            if (this._pendingTimeout) clearTimeout(this._pendingTimeout);
            this._pendingTimeout = setTimeout(() => {
                console.warn('Pending frame timeout reached - clearing pending flag');
                this._pending = false;
            }, 3000); // 3s fallback

            try {
                this._sendCanvas.toBlob((blob) => {
                    if (!blob) {
                        this._pending = false;
                        return;
                    }
                    try {
                        this.ws.send(blob);
                        this._lastSent = Date.now();
                    } catch (err) {
                        console.warn('Failed to send blob over WS', err);
                        this._pending = false;
                        if (this._pendingTimeout) { clearTimeout(this._pendingTimeout); this._pendingTimeout = null; }
                    }
                }, 'image/jpeg', jpegQuality);
            } catch (err) {
                console.warn('toBlob/send error', err);
                this._pending = false;
                if (this._pendingTimeout) { clearTimeout(this._pendingTimeout); this._pendingTimeout = null; }
            }
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
        if (this.aiCameraFeed) this.aiCameraFeed.style.display = show ? 'none' : 'block';
        this.aiStatus.style.display = show ? 'none' : 'flex';
    }

    showError(message = 'Unable to start AI camera. Please check if the AI service is running.') {
        this.cameraError.style.display = 'flex';
        const p = this.cameraError.querySelector('p');
        if (p) p.textContent = message;
        this.cameraLoading.style.display = 'none';
        if (this.aiCameraFeed) this.aiCameraFeed.style.display = 'none';
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
        // Stop MediaPipe processing and animation frame
        try {
            if (this._localHandsRaf) {
                cancelAnimationFrame(this._localHandsRaf);
                this._localHandsRaf = null;
            }
            if (this._hands && typeof this._hands.close === 'function') {
                this._hands.close();
                this._hands = null;
            }
        } catch (err) {
            console.warn('Error cleaning up local hands:', err);
        }
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