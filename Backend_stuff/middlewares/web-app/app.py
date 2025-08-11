"""
SASL Web Application
Complete web-based version of the enhanced camera system
"""

import os
import sys
import json
import cv2
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import numpy as np
from collections import deque
import threading
import time
import base64
from flask import Flask, render_template, Response, jsonify, request
import warnings
warnings.filterwarnings("ignore")

# Suppress MediaPipe verbose logging
os.environ['GLOG_minloglevel'] = '2'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['MEDIAPIPE_DISABLE_GPU'] = '1'

try:
    import mediapipe as mp
    HAND_DETECTION_AVAILABLE = True
except ImportError:
    print("MediaPipe not available. Hand detection disabled.")
    HAND_DETECTION_AVAILABLE = False

# Initialize Flask app
app = Flask(__name__)

# Global variables
camera = None
detector = None
current_prediction = "Waiting..."
current_confidence = 0.0
show_hands = True
frame_count = 0
is_camera_active = False

# Device and paths
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
current_dir = os.path.dirname(os.path.abspath(__file__))

# ============================================================================
# MODEL ARCHITECTURES
# ============================================================================

class HandFocusedCNN_LSTM(nn.Module):
    """Hand-focused CNN-LSTM model for SASL gesture recognition"""
    
    def __init__(self, cnn, hidden_size=256, num_classes=41, num_layers=2, dropout=0.3):
        super(HandFocusedCNN_LSTM, self).__init__()
        self.cnn = cnn
        self.lstm = nn.LSTM(
            input_size=512, 
            hidden_size=hidden_size,
            num_layers=num_layers, 
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        self.dropout = nn.Dropout(dropout)
        
        # Hand attention mechanism
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size * 2,  # 512 for bidirectional
            num_heads=8,
            dropout=dropout,
            batch_first=True
        )
        
        # Classifier module (matches the saved model structure exactly)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, 256),  # classifier.0: [256, 512]
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),              # classifier.3: [128, 256] 
            nn.ReLU(), 
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)       # classifier.6: [num_classes, 128]
        )

    def forward(self, x):  # x: (batch, seq_len, C, H, W)
        batch_size, seq_len, C, H, W = x.size()
        x = x.view(batch_size * seq_len, C, H, W)
        features = self.cnn(x)
        features = features.view(batch_size, seq_len, -1)
        
        # LSTM processing
        lstm_out, _ = self.lstm(features)
        
        # Hand attention
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        combined = lstm_out + attn_out
        
        # Final prediction through classifier
        combined = self.dropout(combined)
        out = self.classifier(combined[:, -1, :])
        return out

def create_cnn_base():
    """Create CNN base network"""
    import torchvision.models as models
    
    # Use ResNet18 as base
    resnet = models.resnet18(weights='IMAGENET1K_V1')
    
    # Remove final layers and add custom ones
    layers = list(resnet.children())[:-2]  # Remove avgpool and fc
    layers.append(nn.AdaptiveAvgPool2d((1, 1)))
    layers.append(nn.Flatten())
    
    return nn.Sequential(*layers)

# ============================================================================
# HAND DETECTION SYSTEM
# ============================================================================

class WebHandDetector:
    """MediaPipe-based hand detection for web application"""
    
    def __init__(self):
        if not HAND_DETECTION_AVAILABLE:
            self.hands = None
            return
            
        self.mp_hands = mp.solutions.hands
        self.mp_draw = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.3
        )
    
    def detect_hands(self, frame):
        """Detect hands in frame and return hand data"""
        if not self.hands:
            return []
            
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)
        
        hands_data = []
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Get bounding box
                h, w, _ = frame.shape
                x_coords = [lm.x * w for lm in hand_landmarks.landmark]
                y_coords = [lm.y * h for lm in hand_landmarks.landmark]
                
                bbox = [
                    int(min(x_coords)),
                    int(min(y_coords)),
                    int(max(x_coords)),
                    int(max(y_coords))
                ]
                
                hands_data.append({
                    'landmarks': hand_landmarks,
                    'bbox': bbox,
                    'confidence': 0.8
                })
        
        return hands_data
    
    def draw_hands(self, frame, hands_data):
        """Draw hand landmarks and bounding boxes"""
        if not self.hands:
            return frame
            
        for hand_data in hands_data:
            # Draw landmarks
            self.mp_draw.draw_landmarks(
                frame, 
                hand_data['landmarks'], 
                self.mp_hands.HAND_CONNECTIONS
            )
            
            # Draw bounding box
            bbox = hand_data['bbox']
            cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
            cv2.putText(frame, f"Hand {hand_data['confidence']:.2f}", 
                       (bbox[0], bbox[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        return frame
    
    def close(self):
        """Clean up resources"""
        if hasattr(self, 'hands') and self.hands:
            self.hands.close()

# ============================================================================
# GESTURE DETECTION SYSTEM
# ============================================================================

class WebGestureDetector:
    """Web-based gesture detection system"""
    
    def __init__(self, model, device, class_names, buffer_size=16):
        self.model = model
        self.device = device
        self.class_names = class_names
        self.buffer_size = buffer_size
        self.frame_buffer = deque(maxlen=buffer_size)
        self.prediction_history = deque(maxlen=10)
        self.hand_detector = WebHandDetector()
        
        # Transform for model input
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    
    def add_frame(self, frame):
        """Add frame to buffer"""
        self.frame_buffer.append(frame.copy())
    
    def is_buffer_ready(self):
        """Check if buffer has enough frames for prediction"""
        return len(self.frame_buffer) >= self.buffer_size
    
    def predict_gesture(self):
        """Predict gesture from current buffer"""
        if not self.is_buffer_ready():
            return None, 0.0
        
        try:
            # Preprocess frames
            frames_tensor = self._preprocess_frames(list(self.frame_buffer))
            frames_tensor = frames_tensor.unsqueeze(0).to(self.device)
            
            # Model prediction
            with torch.no_grad():
                outputs = self.model(frames_tensor)
                
                # Check for NaN outputs
                if torch.isnan(outputs).any() or torch.isinf(outputs).any():
                    print("Model producing NaN outputs")
                    return None, 0.0
                
                probabilities = torch.softmax(outputs, dim=1)
                confidence, predicted_idx = torch.max(probabilities, 1)
                
                predicted_class = self.class_names[predicted_idx.item()]
                confidence_score = confidence.item()
                
                # Filter low confidence predictions
                if confidence_score < 0.01:
                    return None, 0.0
                
                # Add to history
                self.prediction_history.append((predicted_class, confidence_score))
                
                # Get stable prediction
                stable_prediction = self._get_stable_prediction()
                return stable_prediction, confidence_score
                
        except Exception as e:
            print(f"Prediction error: {e}")
            return None, 0.0
    
    def _preprocess_frames(self, frames):
        """Preprocess frames for model input"""
        processed_frames = []
        for frame in frames:
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            tensor_frame = self.transform(frame_rgb)
            processed_frames.append(tensor_frame)
        return torch.stack(processed_frames)
    
    def _get_stable_prediction(self):
        """Get most stable recent prediction"""
        if len(self.prediction_history) < 1:
            return None
        
        # Get recent predictions
        recent_predictions = [pred[0] for pred in list(self.prediction_history)[-3:]]
        
        # Return most common prediction
        prediction_counts = {}
        for pred in recent_predictions:
            prediction_counts[pred] = prediction_counts.get(pred, 0) + 1
        
        most_common = max(prediction_counts.items(), key=lambda x: x[1])
        return most_common[0]
    
    def get_hand_overlay_info(self, frame):
        """Get hand detection information"""
        try:
            return self.hand_detector.detect_hands(frame)
        except Exception as e:
            print(f"Hand detection error: {e}")
            return []
    
    def cleanup(self):
        """Clean up resources"""
        self.hand_detector.close()

# ============================================================================
# MODEL LOADING
# ============================================================================

def validate_model(model, device, class_names):
    """Validate that the model produces valid outputs"""
    try:
        model.eval()
        
        # Create dummy input (batch_size=1, seq_len=16, C=3, H=224, W=224)
        dummy_input = torch.randn(1, 16, 3, 224, 224).to(device)
        
        with torch.no_grad():
            output = model(dummy_input)
            
            # Check output shape
            if output.shape != (1, len(class_names)):
                print(f"Model output shape mismatch: expected (1, {len(class_names)}), got {output.shape}")
                return False
            
            # Check for NaN or infinite values
            if torch.isnan(output).any() or torch.isinf(output).any():
                print("Model produces NaN or infinite values")
                return False
            
            # Apply softmax and check probabilities
            probs = torch.softmax(output, dim=1)
            if torch.isnan(probs).any() or torch.isinf(probs).any():
                print("Softmax produces NaN or infinite values")
                return False
            
            print("✓ Model validation passed")
            return True
            
    except Exception as e:
        print(f"Model validation failed: {e}")
        return False

def create_model_from_checkpoint(checkpoint, class_names, device):
    """Create model architecture that matches the checkpoint structure"""
    
    # Analyze classifier structure from checkpoint
    classifier_layers = []
    layer_idx = 0
    
    while f'classifier.{layer_idx}.weight' in checkpoint:
        weight_shape = checkpoint[f'classifier.{layer_idx}.weight'].shape
        bias_shape = checkpoint[f'classifier.{layer_idx}.bias'].shape
        
        print(f"  Classifier layer {layer_idx}: Linear({weight_shape[1]} -> {weight_shape[0]})")
        classifier_layers.append((weight_shape[1], weight_shape[0]))
        layer_idx += 3  # Skip ReLU and Dropout layers
    
    # Create model with dynamic classifier
    class DynamicHandFocusedCNN_LSTM(nn.Module):
        def __init__(self, cnn, hidden_size=256, num_classes=41, num_layers=2, dropout=0.3):
            super(DynamicHandFocusedCNN_LSTM, self).__init__()
            self.cnn = cnn
            self.lstm = nn.LSTM(
                input_size=512, 
                hidden_size=hidden_size,
                num_layers=num_layers, 
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0,
                bidirectional=True
            )
            self.dropout = nn.Dropout(dropout)
            
            # Hand attention mechanism
            self.attention = nn.MultiheadAttention(
                embed_dim=hidden_size * 2,
                num_heads=8,
                dropout=dropout,
                batch_first=True
            )
            
            # Dynamic classifier based on checkpoint structure
            classifier_modules = []
            for i, (in_features, out_features) in enumerate(classifier_layers):
                classifier_modules.append(nn.Linear(in_features, out_features))
                if i < len(classifier_layers) - 1:  # Don't add ReLU/Dropout after last layer
                    classifier_modules.append(nn.ReLU())
                    classifier_modules.append(nn.Dropout(dropout))
            
            self.classifier = nn.Sequential(*classifier_modules)
            
            print(f"✓ Created dynamic classifier with {len(classifier_layers)} linear layers")

        def forward(self, x):
            batch_size, seq_len, C, H, W = x.size()
            x = x.view(batch_size * seq_len, C, H, W)
            features = self.cnn(x)
            features = features.view(batch_size, seq_len, -1)
            
            # LSTM processing
            lstm_out, _ = self.lstm(features)
            
            # Hand attention
            attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
            combined = lstm_out + attn_out
            
            # Final prediction through dynamic classifier
            combined = self.dropout(combined)
            out = self.classifier(combined[:, -1, :])
            return out
    
    # Create CNN base
    cnn_base = create_cnn_base()
    
    # Create model with dynamic classifier
    model = DynamicHandFocusedCNN_LSTM(
        cnn=cnn_base,
        num_classes=len(class_names),
        hidden_size=256,
        num_layers=2,
        dropout=0.3
    ).to(device)
    
    return model

def load_model_and_classes():
    """Load the SASL model and class names"""
    # Load class names
    class_names_path = os.path.join(current_dir, "models", "class_names.json")
    try:
        with open(class_names_path, 'r') as f:
            class_names = json.load(f)
        print(f"✓ Loaded {len(class_names)} classes")
    except Exception as e:
        print(f"Error loading class names: {e}")
        return None, None
    
    # Load model
    model_path = os.path.join(current_dir, "models", "hand_focused_sasl_model.pth")
    if not os.path.exists(model_path):
        print(f"Model file not found: {model_path}")
        print("Please ensure 'hand_focused_sasl_model.pth' is in the models/ directory")
        return None, None
    
    try:
        # Create model architecture
        cnn_base = create_cnn_base()
        model = HandFocusedCNN_LSTM(
            cnn=cnn_base,
            num_classes=len(class_names),
            hidden_size=256,
            num_layers=2,
            dropout=0.3
        ).to(device)
        
        print(f"✓ Model architecture created")
        
        # Load weights
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        
        # Check if the checkpoint keys match our model
        model_keys = set(model.state_dict().keys())
        checkpoint_keys = set(checkpoint.keys())
        
        missing_keys = model_keys - checkpoint_keys
        unexpected_keys = checkpoint_keys - model_keys
        
        if missing_keys:
            print(f"Missing keys in checkpoint: {missing_keys}")
            return None, None
        
        if unexpected_keys:
            print(f"! Unexpected keys in checkpoint: {unexpected_keys}")
        
        model.load_state_dict(checkpoint, strict=False)
        model.eval()
        
        print(f"✓ Model weights loaded successfully")
        
        # Validate the model
        if not validate_model(model, device, class_names):
            print(f"Model validation failed")
            return None, None
        
        print(f"✓ Model loaded and validated on {device}")
        return model, class_names
        
    except Exception as e:
        print(f"Error loading model: {e}")
        import traceback
        traceback.print_exc()
        return None, None

# Initialize model and detector
model, class_names = load_model_and_classes()
if model is None or class_names is None:
    print("Failed to load model or class names. Exiting.")
    sys.exit(1)

detector = WebGestureDetector(model, device, class_names)

# ============================================================================
# VIDEO STREAMING
# ============================================================================

def generate_frames():
    """Generate video frames for web streaming"""
    global current_prediction, current_confidence, frame_count, camera, is_camera_active
    
    while is_camera_active and camera is not None:
        success, frame = camera.read()
        if not success:
            break
        
        frame_count += 1
        
        # Add frame to detector
        detector.add_frame(frame)
        
        # Get prediction every few frames
        if frame_count % 3 == 0 and detector.is_buffer_ready():
            pred, conf = detector.predict_gesture()
            if pred:
                current_prediction = pred
                current_confidence = conf
        
        # Draw overlays
        frame = draw_overlays(frame)
        
        # Encode frame
        ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if ret:
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

def draw_overlays(frame):
    """Draw all overlays on frame"""
    global show_hands
    
    # Hand detection overlay
    if show_hands:
        hands_data = detector.get_hand_overlay_info(frame)
        if hands_data:
            frame = detector.hand_detector.draw_hands(frame, hands_data)
    
    # Status overlays
    h, w = frame.shape[:2]
    
    # Buffer status
    if not detector.is_buffer_ready():
        buffer_text = f"Collecting frames: {len(detector.frame_buffer)}/{detector.buffer_size}"
        cv2.putText(frame, buffer_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    
    # Prediction
    if current_prediction != "Waiting...":
        # Confidence color coding
        if current_confidence > 0.7:
            color = (0, 255, 0)  # Green
        elif current_confidence > 0.3:
            color = (0, 165, 255)  # Orange
        else:
            color = (0, 0, 255)  # Red
        
        pred_text = f"Gesture: {current_prediction} ({current_confidence:.3f})"
        cv2.putText(frame, pred_text, (10, h-100), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
    
    # Model info
    cv2.putText(frame, "Model: Hand-Focused SASL", (10, h-70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(frame, "Hand-focused attention active", (10, h-45), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    
    # Hand detection status
    hand_status = "ON" if show_hands else "OFF"
    cv2.putText(frame, f"Hands: {hand_status}", (w-120, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    return frame

# ============================================================================
# WEB ROUTES
# ============================================================================

@app.route('/')
def index():
    """Main page"""
    return render_template('index.html', class_names=class_names)

@app.route('/video_feed')
def video_feed():
    """Video streaming route"""
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/start_camera', methods=['POST'])
def start_camera():
    """Start camera capture"""
    global camera, is_camera_active, current_prediction, current_confidence
    
    try:
        camera = cv2.VideoCapture(0)
        if not camera.isOpened():
            return jsonify({'status': 'error', 'message': 'Could not open camera'})
        
        # Set camera properties
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        camera.set(cv2.CAP_PROP_FPS, 30)
        
        is_camera_active = True
        current_prediction = "Camera started - collecting frames..."
        current_confidence = 0.0
        
        return jsonify({'status': 'success', 'message': 'Camera started'})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/stop_camera', methods=['POST'])
def stop_camera():
    """Stop camera capture"""
    global camera, is_camera_active, current_prediction, current_confidence
    
    is_camera_active = False
    if camera:
        camera.release()
        camera = None
    
    current_prediction = "Camera stopped"
    current_confidence = 0.0
    
    return jsonify({'status': 'success', 'message': 'Camera stopped'})

@app.route('/toggle_hands', methods=['POST'])
def toggle_hands():
    """Toggle hand detection overlay"""
    global show_hands
    show_hands = not show_hands
    return jsonify({'status': 'success', 'show_hands': show_hands})

@app.route('/reset_detector', methods=['POST'])
def reset_detector():
    """Reset the gesture detector"""
    global current_prediction, current_confidence, frame_count
    detector.frame_buffer.clear()
    detector.prediction_history.clear()
    current_prediction = "Buffer reset - collecting frames..." if is_camera_active else "Camera stopped"
    current_confidence = 0.0
    frame_count = 0
    return jsonify({'status': 'success'})

@app.route('/status')
def status():
    """Get current status"""
    return jsonify({
        'prediction': current_prediction,
        'confidence': current_confidence,
        'show_hands': show_hands,
        'buffer_ready': detector.is_buffer_ready(),
        'buffer_size': len(detector.frame_buffer),
        'is_camera_active': is_camera_active
    })

# ============================================================================
# MAIN APPLICATION
# ============================================================================

if __name__ == '__main__':
    print("🤟 SASL Web Application 🤟")
    print("=" * 50)
    print(f"Device: {device}")
    print(f"Classes: {len(class_names)}")
    print(f"Hand Detection: {'Available' if HAND_DETECTION_AVAILABLE else 'Disabled'}")
    print("\nStarting web server...")
    print("Open your browser and go to: http://localhost:5000")
    print("Press Ctrl+C to stop the server")
    
    try:
        app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        if camera:
            camera.release()
        if detector:
            detector.cleanup()
        cv2.destroyAllWindows()