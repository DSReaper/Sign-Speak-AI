"""
SASL Web Application - Flask AI Backend
Provides AI gesture recognition endpoints for Node.js frontend
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
from flask import Flask, render_template, jsonify, request
from flask_cors import CORS
import warnings
warnings.filterwarnings("ignore")
import asyncio
import websockets
from io import BytesIO
from queue import Queue, Empty
from datetime import datetime
import timm
import glob
import joblib

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
CORS(app)  # Enable CORS for Node.js integration

# Global variables
detector = None
alphabet_detector = None  # Hand landmark model for alphabet mode
current_model_mode = 'motion'  # 'motion' or 'alphabet'
model_mode_lock = threading.Lock()
current_prediction = "Loading the AI detection model..."
current_confidence = 0.0
# Grammar / sentence construction session state
recognized_words = []  # list of dicts: {text, confidence, t_utc}
last_display_word = None
last_display_start = None
last_committed_word = None
COMMIT_SECONDS = 0.5  # hold duration before committing a stable prediction
show_hands = True
# When False, the server will not draw status/model/prediction text overlays
# on the returned image frames. Hand landmark/box overlays remain controlled
# separately by `show_hands` so the visual hand guidance is preserved.
show_server_overlays = False

frame_count = 0
detector_lock = threading.Lock()

# Import grammar utilities (rule-based grammar)
try:  # packaged execution
    from .grammar_utils import grammar_fix  # type: ignore
except Exception:
    try:  # script-style execution
        from grammar_utils import grammar_fix  # type: ignore
    except Exception:
        def grammar_fix(words):  # type: ignore
            return " ".join(words).strip()
        print("WARNING: grammar_utils.py was not found.")

# Device and paths
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
current_dir = os.path.dirname(os.path.abspath(__file__))

# ============================================================================
# MODEL ARCHITECTURES 
# ============================================================================

class CNNLSTMModel(nn.Module):
    """CNN+LSTM model for video classification using PyTorch"""
    
    def __init__(self, num_classes, sequence_length=30, input_size=(224, 224)):
        super(CNNLSTMModel, self).__init__()
        
        self.sequence_length = sequence_length
        self.input_size = input_size
        self.num_classes = num_classes
        
        # Pre-trained CNN backbone (EfficientNet)
        self.backbone = timm.create_model('efficientnet_b0', pretrained=True, num_classes=0)
        
        # Freeze backbone for transfer learning
        for param in self.backbone.parameters():
            param.requires_grad = False
        
        # Get feature dimension from backbone
        feature_dim = self.backbone.num_features
        
        # Temporal processing layers
        self.temporal_conv = nn.Conv1d(feature_dim, 512, kernel_size=3, padding=1)
        self.temporal_bn = nn.BatchNorm1d(512)
        self.dropout1 = nn.Dropout(0.3)
        
        # LSTM layers
        self.lstm1 = nn.LSTM(512, 256, bidirectional=True, batch_first=True)
        self.lstm2 = nn.LSTM(512, 128, bidirectional=True, batch_first=True)
        self.dropout_lstm = nn.Dropout(0.3)
        
        # Classification layers
        self.classifier = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        batch_size, seq_len, c, h, w = x.size()
        
        # Process each frame through CNN
        x = x.view(-1, c, h, w)  # (batch*seq, c, h, w)
        features = self.backbone(x)  # (batch*seq, feature_dim)
        
        # Reshape back to sequence
        features = features.view(batch_size, seq_len, -1)  # (batch, seq, feature_dim)
        
        # Temporal convolution
        x = features.transpose(1, 2)  # (batch, feature_dim, seq)
        x = torch.relu(self.temporal_bn(self.temporal_conv(x)))
        x = self.dropout1(x)
        x = x.transpose(1, 2)  # (batch, seq, 512)
        
        # LSTM layers
        x, _ = self.lstm1(x)  # (batch, seq, 512)
        x = self.dropout_lstm(x)
        x, _ = self.lstm2(x)  # (batch, seq, 256)
        
        # Global average pooling over sequence
        x = torch.mean(x, dim=1)  # (batch, 256)
        
        # Classification
        x = self.classifier(x)
        
        return x



# ============================================================================
# HAND LANDMARK MODEL (ALPHABET MODE)
# ============================================================================

class HandGestureRecognizer:
    """Hand landmark-based gesture recognition for alphabet mode (single-frame)."""
    
    def __init__(self, model_dir):
        """
        Initialize the recognizer with a trained hand landmark model.
        
        Args:
            model_dir: Directory containing the trained model files
        """
        print(f"Loading alphabet model from: {model_dir}")
        
        # Load model components
        self.model = joblib.load(f"{model_dir}/random_forest_model.joblib")
        self.scaler = joblib.load(f"{model_dir}/scaler.joblib")
        self.label_encoder = joblib.load(f"{model_dir}/label_encoder.joblib")
        
        print(f"Alphabet model loaded successfully!")
        print(f"Recognizing {len(self.label_encoder.classes_)} gestures: {', '.join(self.label_encoder.classes_)}")
        
        # Initialize MediaPipe Hands
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # For smoothing predictions
        self.prediction_history = deque(maxlen=5)
        
        # Check model input size
        expected_features = self.scaler.n_features_in_
        self.two_hand_mode = (expected_features == 126)
        
        if self.two_hand_mode:
            print("✓ Alphabet model supports TWO-HAND gestures (left + right)")
        else:
            print("✓ Alphabet model uses SINGLE-HAND format")
        
    def extract_landmarks(self, hand_landmarks):
        """Extract landmark coordinates from MediaPipe hand landmarks."""
        landmarks = []
        for landmark in hand_landmarks.landmark:
            landmarks.extend([landmark.x, landmark.y, landmark.z])
        return np.array(landmarks)
    
    def predict_from_frame(self, frame):
        """
        Predict gesture from a single frame (no buffering).
        
        Args:
            frame: OpenCV BGR image
            
        Returns:
            predicted_class: Predicted gesture class (or None)
            confidence: Prediction confidence
        """
        # Convert to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)
        
        if not results.multi_hand_landmarks:
            return None, 0.0
        
        # Collect hand data
        hands_data = {}
        for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
            hand_label = handedness.classification[0].label.lower()
            hands_data[hand_label] = self.extract_landmarks(hand_landmarks)
        
        # Prepare feature vector
        if self.two_hand_mode:
            left_landmarks = hands_data.get('left', np.zeros(63))
            right_landmarks = hands_data.get('right', np.zeros(63))
            features = np.concatenate([left_landmarks, right_landmarks]).reshape(1, -1)
        else:
            if 'left' in hands_data:
                features = hands_data['left'].reshape(1, -1)
            elif 'right' in hands_data:
                features = hands_data['right'].reshape(1, -1)
            else:
                return None, 0.0
        
        # Scale and predict
        features_scaled = self.scaler.transform(features)
        prediction = self.model.predict(features_scaled)
        probabilities = self.model.predict_proba(features_scaled)[0]
        
        predicted_class = self.label_encoder.inverse_transform(prediction)[0]
        confidence = np.max(probabilities)
        
        # Add to history for smoothing
        self.prediction_history.append(predicted_class)
        
        # Use most common prediction in recent history
        if len(self.prediction_history) >= 3:
            from collections import Counter
            smoothed_prediction = Counter(self.prediction_history).most_common(1)[0][0]
            return smoothed_prediction, confidence
        
        return predicted_class, confidence
    
    def cleanup(self):
        """Clean up resources"""
        if hasattr(self, 'hands') and self.hands:
            self.hands.close()


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

def find_latest_training_folder():
    """Find the latest training folder, preferring outputs/ over models/.

    Search order (both relative to this file's directory):
      1) outputs/training_*
      2) models/training_*
    """
    outputs_dir = os.path.join(current_dir, "outputs")
    models_dir = os.path.join(current_dir, "models")

    # Prefer outputs directory first
    outputs_training = glob.glob(os.path.join(outputs_dir, "training_*"))
    models_training = glob.glob(os.path.join(models_dir, "training_*"))

    candidates = []
    if outputs_training:
        candidates.extend(outputs_training)
    if models_training:
        candidates.extend(models_training)

    if not candidates:
        return None

    # Choose the lexicographically latest (timestamp in name)
    latest_folder = max(candidates)
    return latest_folder

def validate_model(model, device, class_names):
    """Validate that the model produces valid outputs"""
    try:
        model.eval()
        
        # Create dummy input using model's preferred sequence length if available
        seq_len = getattr(model, 'sequence_length', 16)
        # (batch_size=1, seq_len, C=3, H=224, W=224)
        dummy_input = torch.randn(1, seq_len, 3, 224, 224).to(device)
        
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
            
            print("Model validation passed")
            return True
            
    except Exception as e:
        print(f"Model validation failed: {e}")
        return False

def load_model_and_classes():
    """Load the SASL model and class names from training folder"""
    
    # Find the latest training folder (prefer outputs/ over models/)
    training_folder = find_latest_training_folder()
    if not training_folder:
        print("ERROR: No training folder found in outputs/ or models/ directory.")
        print("Please ensure you have a training folder (e.g., training_20251011_163502) with:")
        print("  - models/best_sasl_cnn_lstm_model.pth")
        print("  - results/class_names.json")
        return None, None
    
    print(f"Using training folder: {os.path.basename(training_folder)}")
    
    # Load class names from results folder
    class_names_path = os.path.join(training_folder, "results", "class_names.json")
    try:
        with open(class_names_path, 'r', encoding='utf-8') as f:
            class_names = json.load(f)
        print(f"Loaded {len(class_names)} classes from results folder")
    except Exception as e:
        print(f"ERROR: Failed to load class names from: {class_names_path}")
        print(f"Error: {e}")
        return None, None
    
    # Load model from models folder within training directory
    model_path = os.path.join(training_folder, "models", "best_sasl_cnn_lstm_model.pth")
    if not os.path.exists(model_path):
        print(f"ERROR: Model file not found: {model_path}")
        print("Please ensure the model file exists in the training folder.")
        return None, None
    
    try:
        # Create CNN+LSTM model architecture
        model = CNNLSTMModel(
            num_classes=len(class_names),
            sequence_length=30,
            input_size=(224, 224)
        ).to(device)
        
        print(f"CNN+LSTM model architecture created")
        
        # Load weights
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        
        # Check if the checkpoint keys match our model
        model_keys = set(model.state_dict().keys())
        checkpoint_keys = set(checkpoint.keys())
        
        missing_keys = model_keys - checkpoint_keys
        unexpected_keys = checkpoint_keys - model_keys
        
        if missing_keys:
            print(f"ERROR: Missing keys in checkpoint: {len(missing_keys)} keys missing")
            print("The model architecture may not match the saved checkpoint.")
            return None, None
        
        if unexpected_keys:
            print(f"WARNING: Unexpected keys in checkpoint: {len(unexpected_keys)} extra keys")
            print("This may indicate a model version mismatch but loading will continue.")
        
        model.load_state_dict(checkpoint, strict=False)
        model.eval()
        
        print(f"Model weights loaded successfully")
        
        # Validate the model
        if not validate_model(model, device, class_names):
            print(f"ERROR: Model validation failed")
            print("The loaded model does not produce valid outputs.")
            return None, None
        
        print(f"CNN+LSTM model loaded and validated on {device}")
        return model, class_names
        
    except Exception as e:
        print(f"ERROR: Failed to load model: {e}")
        import traceback
        traceback.print_exc()
        return None, None


def find_latest_hand_landmark_model(base_dir="models"):
    """
    Find the most recent hand landmark model directory for alphabet mode.
    
    Args:
        base_dir: Base directory to search for models
        
    Returns:
        Path to the latest hand landmark model directory, or None if not found
    """
    pattern = os.path.join(current_dir, base_dir, "hand_landmark_model_*")
    model_dirs = glob.glob(pattern)
    
    if not model_dirs:
        return None
    
    # Sort by directory name (which includes timestamp) and get the latest
    model_dirs.sort(reverse=True)
    return model_dirs[0]


def load_alphabet_model():
    """Load the hand landmark model for alphabet mode"""
    
    # Find the latest hand landmark model
    model_path = find_latest_hand_landmark_model()
    if model_path is None:
        print("Warning: No hand landmark model found for alphabet mode")
        print("Alphabet mode will be disabled")
        return None
    
    print(f"Using alphabet model: {os.path.basename(model_path)}")
    
    try:
        # Check if required files exist
        required_files = ['random_forest_model.joblib', 'scaler.joblib', 'label_encoder.joblib']
        for fname in required_files:
            fpath = os.path.join(model_path, fname)
            if not os.path.exists(fpath):
                print(f"Error: Required file not found: {fname}")
                return None
        
        # Initialize the recognizer
        recognizer = HandGestureRecognizer(model_path)
        print("Alphabet model loaded and ready")
        return recognizer
        
    except Exception as e:
        print(f"ERROR: Failed to load alphabet model: {e}")
        import traceback
        traceback.print_exc()
        return None



# Initialize models and detectors
print("="*60)
print("Loading SASL AI Models...")
print("="*60)

# Load motion model (CNN-LSTM)
print("\n[1/2] Loading Motion Model (CNN-LSTM)...")
model, class_names = load_model_and_classes()
if model is None or class_names is None:
    print("Failed to load motion model. Motion mode will be disabled.")
    detector = None
else:
    # Use a 30-frame buffer to align with typical training sequence length
    detector = WebGestureDetector(model, device, class_names, buffer_size=30)
    print("✓ Motion model loaded successfully")

# Load alphabet model (Hand Landmarks)
print("\n[2/2] Loading Alphabet Model (Hand Landmarks)...")
alphabet_detector = load_alphabet_model()
if alphabet_detector:
    print("✓ Alphabet model loaded successfully")
else:
    print("⚠ Alphabet model not available")

print("\n" + "="*60)
print("Model Loading Complete")
print(f"Motion Mode: {'Available' if detector else 'Disabled'}")
print(f"Alphabet Mode: {'Available' if alphabet_detector else 'Disabled'}")
print("="*60 + "\n")

# ============================================================================
# VIDEO STREAMING
# ============================================================================

# The application now treats web client frames (sent over WebSocket) as the input source. 
# See ws_handler and FrameProcessorWorker below.

def draw_overlays(frame):
    """Draw all overlays on frame"""
    global show_hands
    global show_server_overlays
    
    # Hand detection overlay (use lock when interacting with detector)
    if show_hands and detector:
        with detector_lock:
            hands_data = detector.get_hand_overlay_info(frame)
            if hands_data:
                # Draw hands with slightly thicker visuals for visibility
                frame = detector.hand_detector.draw_hands(frame, hands_data)
                try:
                    # Also draw bounding boxes more prominently
                    for hand_data in hands_data:
                        bbox = hand_data.get('bbox', None)
                        if bbox and len(bbox) == 4:
                            cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 3)
                except Exception:
                    pass
    
    # Status/model/prediction overlays are optional. If the server flag
    # `show_server_overlays` is False we skip drawing those textual overlays so
    # the image contains only the visual hand guidance (if enabled).
    if show_server_overlays:
        # Status overlays
        h, w = frame.shape[:2]

        # Buffer status
        try:
            if detector:
                buffer_text = f"Frames: {len(detector.frame_buffer)}/{detector.buffer_size}"
            else:
                buffer_text = "Detector: not loaded"
            # Draw a background for readability
            (tx, ty), _ = cv2.getTextSize(buffer_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            cv2.rectangle(frame, (10, 10), (10 + tx + 12, 10 + ty + 12), (0, 0, 0), -1)
            cv2.putText(frame, buffer_text, (16, 10 + ty + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        except Exception:
            pass

        # Prediction
        try:
            pred_text = None
            if current_prediction and current_prediction not in ["Waiting...", "Loading the AI detection model..."]:
                # Confidence color coding
                if current_confidence > 0.7:
                    color = (0, 220, 0)  # Green
                elif current_confidence > 0.3:
                    color = (0, 165, 255)  # Orange
                else:
                    color = (0, 0, 255)  # Red
                pred_text = f"{current_prediction} ({current_confidence:.3f})"
            else:
                pred_text = "Waiting for prediction..."

            # Draw prediction box centered near bottom
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.9
            thickness = 2
            (pw, ph), _ = cv2.getTextSize(pred_text, font, font_scale, thickness)
            box_w = pw + 24
            box_h = ph + 18
            box_x = max(10, (w - box_w) // 2)
            box_y = h - box_h - 10
            # Semi-opaque background
            overlay = frame.copy()
            cv2.rectangle(overlay, (box_x, box_y), (box_x + box_w, box_y + box_h), (0, 0, 0), -1)
            alpha = 0.6
            cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
            # Text
            text_x = box_x + 12
            text_y = box_y + box_h - 8
            cv2.putText(frame, pred_text, (text_x, text_y), font, font_scale, color, thickness, cv2.LINE_AA)
        except Exception:
            pass

        # Model info
        if detector:
            try:
                cv2.putText(frame, "Model: Hand-Focused SASL", (10, h-70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(frame, "Hand-focused attention active", (10, h-45), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
            except Exception:
                pass
        else:
            try:
                cv2.putText(frame, "AI Model: Not loaded", (10, h-45), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            except Exception:
                pass

        # Hand detection status
        try:
            hand_status = "ON" if show_hands else "OFF"
            status_text = f"Hands: {hand_status}"
            (sx, sy), _ = cv2.getTextSize(status_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(frame, (w - sx - 22, 10), (w - 10, 10 + sy + 12), (0, 0, 0), -1)
            cv2.putText(frame, status_text, (w - sx - 16, 10 + sy + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        except Exception:
            pass
    
    return frame

# ============================================================================
# WEB ROUTES (Updated for Node.js integration)
# ============================================================================

# Clients send frames over the WebSocket for detection and receive JSON responses.

@app.route('/switch_mode', methods=['POST'])
def switch_mode():
    """Switch between motion and alphabet detection modes"""
    global current_model_mode, detector, alphabet_detector
    global recognized_words, last_display_word, last_display_start, last_committed_word
    
    try:
        data = request.get_json()
        mode = data.get('mode', 'motion')
        
        if mode not in ['motion', 'alphabet']:
            return jsonify({'status': 'error', 'message': 'Invalid mode. Use "motion" or "alphabet"'}), 400
        
        with model_mode_lock:
            # Check if requested model is available
            if mode == 'motion' and detector is None:
                return jsonify({'status': 'error', 'message': 'Motion model not available'}), 503
            if mode == 'alphabet' and alphabet_detector is None:
                return jsonify({'status': 'error', 'message': 'Alphabet model not available'}), 503
            
            # Clear any buffered state when switching modes
            if mode == 'motion' and detector:
                with detector_lock:
                    detector.frame_buffer.clear()
                    detector.prediction_history.clear()
            elif mode == 'alphabet' and alphabet_detector:
                alphabet_detector.prediction_history.clear()
            
            # Reset sentence construction state
            recognized_words = []
            last_display_word = None
            last_display_start = None
            last_committed_word = None
            
            # Switch mode
            old_mode = current_model_mode
            current_model_mode = mode
            
            print(f"Switched from {old_mode} mode to {mode} mode")
            
            return jsonify({
                'status': 'success',
                'mode': current_model_mode,
                'message': f'Switched to {mode} mode'
            })
    except Exception as e:
        print(f"Error switching mode: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

# The frontend captures frames and sends them to the WebSocket server.

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
    global recognized_words, last_display_word, last_display_start, last_committed_word
    
    if detector:
        detector.frame_buffer.clear()
        detector.prediction_history.clear()
    
    current_prediction = "Buffer reset - collecting frames..."
    current_confidence = 0.0
    frame_count = 0
    recognized_words.clear()
    last_display_word = None
    last_display_start = None
    last_committed_word = None
    return jsonify({'status': 'success'})

@app.route('/status')
def status():
    """Get current status"""
    committed = [w['text'] for w in recognized_words]
    sentence = grammar_fix(committed)
    return jsonify({
        'prediction': current_prediction,
        'confidence': current_confidence,
        'mode': current_model_mode,
        'motion_available': detector is not None,
        'alphabet_available': alphabet_detector is not None,
        'show_hands': show_hands,
        'show_server_overlays': show_server_overlays,
        'buffer_ready': detector.is_buffer_ready() if detector else False,
        'buffer_size': len(detector.frame_buffer) if detector else 0,
        'model_loaded': detector is not None,
        'committed_words': committed,
        'sentence': sentence,
    })


@app.route('/toggle_overlays', methods=['POST'])
def toggle_overlays():
    """Toggle whether the server draws status/prediction/model overlays onto frames."""
    global show_server_overlays
    show_server_overlays = not show_server_overlays
    return jsonify({'status': 'success', 'show_server_overlays': show_server_overlays})

@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'ai_model': 'loaded' if detector else 'not_loaded',
        'input_source': 'web_client_frames',
        'device': str(device)
    })


# ============================================================================
# WEBSOCKET SERVER FOR LOW-LATENCY FRAME PROCESSING
# Clients send binary JPEG frames; server processes them and returns small JSON
# messages containing prediction metadata (prediction + confidence). 
# ============================================================================

async def ws_handler(websocket):
    class FrameProcessorWorker:
        def __init__(self, ws, loop, queue_maxsize=1):
            self.ws = ws
            self.loop = loop
            self.queue = Queue(maxsize=queue_maxsize)
            self._stop_event = threading.Event()
            self.thread = threading.Thread(target=self._run, daemon=True)
            self.thread.start()

        def push_frame(self, frame_bytes):
            # Non-blocking: drop older frame when queue full to keep latest only
            try:
                if self.queue.full():
                    try:
                        self.queue.get_nowait()
                    except Empty:
                        pass
                self.queue.put_nowait(frame_bytes)
            except Exception as e:
                # If anything goes wrong, just drop the frame
                print(f"FrameProcessorWorker: push_frame drop due to {e}")

        def stop(self):
            self._stop_event.set()
            # Put a sentinel to unblock the thread if waiting
            try:
                self.queue.put_nowait(None)
            except Exception:
                pass
            self.thread.join(timeout=1.0)

        def _run(self):
            while not self._stop_event.is_set():
                try:
                    item = self.queue.get(timeout=0.5)
                except Empty:
                    continue
                if item is None:
                    break
                try:
                    # Run CPU-bound processing synchronously in this thread
                    result = process_frame_bytes_sync(item)
                    if result:
                        # result is a dict containing prediction metadata
                        try:
                            fut = asyncio.run_coroutine_threadsafe(self.ws.send(json.dumps(result)), self.loop)
                            try:
                                fut.result(timeout=3.0)
                            except Exception as send_exc:
                                print(f"FrameProcessorWorker: send failed: {send_exc}")
                        except Exception as sch_exc:
                            print(f"FrameProcessorWorker: schedule send failed: {sch_exc}")
                except Exception as e:
                    print(f"FrameProcessorWorker: processing error: {e}")

    try:
        # websockets library newer versions provide the path on the websocket object
        ws_path = getattr(websocket, 'path', None)
        print(f"WS client connected: {websocket.remote_address} path={ws_path}")

        # Create a per-connection worker and tie it to this websocket's loop
        loop = asyncio.get_event_loop()
        worker = FrameProcessorWorker(websocket, loop, queue_maxsize=1)

        async for message in websocket:
            try:
                if isinstance(message, bytes):
                    # Enqueue the frame for background processing; return quickly
                    worker.push_frame(message)
                else:
                    # Text messages for mode switching or commands
                    try:
                        cmd = json.loads(message)
                        print(f"WS: Received command: {cmd}")
                        
                        if cmd.get('action') == 'switch_mode':
                            new_mode = cmd.get('mode', 'motion')
                            print(f"WS: Mode switch request to: {new_mode}")
                            
                            if new_mode in ['motion', 'alphabet']:
                                with model_mode_lock:
                                    global current_model_mode, recognized_words
                                    global last_display_word, last_display_start, last_committed_word
                                    
                                    # Check availability
                                    if new_mode == 'motion' and detector is None:
                                        error_msg = json.dumps({'error': 'Motion model not available', 'status': 'error'})
                                        print(f"WS: {error_msg}")
                                        await websocket.send(error_msg)
                                    elif new_mode == 'alphabet' and alphabet_detector is None:
                                        error_msg = json.dumps({'error': 'Alphabet model not available', 'status': 'error'})
                                        print(f"WS: {error_msg}")
                                        await websocket.send(error_msg)
                                    else:
                                        # Clear state
                                        if new_mode == 'motion' and detector:
                                            with detector_lock:
                                                detector.frame_buffer.clear()
                                                detector.prediction_history.clear()
                                        elif new_mode == 'alphabet' and alphabet_detector:
                                            alphabet_detector.prediction_history.clear()
                                        
                                        recognized_words = []
                                        last_display_word = None
                                        last_display_start = None
                                        last_committed_word = None
                                        
                                        old_mode = current_model_mode
                                        current_model_mode = new_mode
                                        print(f"WS: Successfully switched from {old_mode} to {new_mode} mode")
                                        
                                        success_msg = json.dumps({
                                            'status': 'success',
                                            'mode': current_model_mode
                                        })
                                        print(f"WS: Sending response: {success_msg}")
                                        await websocket.send(success_msg)
                            else:
                                error_msg = json.dumps({'error': 'Invalid mode', 'status': 'error'})
                                print(f"WS: {error_msg}")
                                await websocket.send(error_msg)
                        else:
                            print(f"WS: Unknown command, sending OK")
                            await websocket.send('OK')
                    except json.JSONDecodeError as e:
                        text_preview = str(message)[:200]
                        print(f"WS: JSON decode error: {e}, message: {text_preview}")
                        await websocket.send('OK')
            except websockets.exceptions.ConnectionClosed:
                print(f"WS loop connection closed while handling message from {websocket.remote_address}")
                break
            except Exception as inner_e:
                print(f"WS handler inner exception: {inner_e}")
                import traceback as _tb
                _tb.print_exc()
                # Try to send an error notice to client (best-effort)
                try:
                    await websocket.send('ERROR')
                except Exception:
                    pass
    except websockets.exceptions.ConnectionClosed as cc:
        print(f"WS connection closed prematurely: {cc}")
    except Exception as e:
        print(f"WS error (outer): {e}")
        import traceback as _tb
        _tb.print_exc()
    finally:
        try:
            # Stop worker and drain resources
            try:
                worker.stop()
            except Exception:
                pass
            ws_path = getattr(websocket, 'path', None)
            print(f"WS client disconnected: {websocket.remote_address} path={ws_path}")
        except Exception:
            print("WS client disconnected (remote address unavailable)")


# Helper sync wrapper because cv2 and torch code is blocking and easier to run in threadpool
def process_frame_bytes_sync(frame_bytes):
    """Synchronous wrapper for processing incoming JPEG bytes.

    This version treats incoming frames from the web client as the primary
    input and returns a JSON-serializable dict containing prediction metadata.
    Supports both motion mode (CNN-LSTM with buffering) and alphabet mode (single-frame).

    Returns:
    - dict with keys: 'prediction' (str or None), 'confidence' (float), 'mode' (str)
    """
    # Ensure globals are declared before any reference within this function
    global current_prediction, current_confidence
    global last_display_word, last_display_start, last_committed_word
    global recognized_words, current_model_mode
  # The json object that will be sent back to the client
    result = {
        'prediction': None,
        'confidence': 0.0,
        'committed_words': [],
        'sentence': "",
        'mode': current_model_mode,
    }

    try:
        # Decode bytes and log basic diagnostics
        nparr = np.frombuffer(frame_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            print("process_frame_bytes_sync: cv2.imdecode failed for incoming frame")
            return result

        # Mirror the frame horizontally to match user perspective
        img = cv2.flip(img, 1)

        # optional debug metric
        try:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            mean_brightness = float(np.mean(gray))
        except Exception:
            mean_brightness = -1.0

        # Process frame based on current mode
        with model_mode_lock:
            mode = current_model_mode
        
        # Debug: log mode every 30 frames
        global frame_count
        if frame_count % 30 == 0:
            print(f"[FRAME] Processing in {mode.upper()} mode (frame {frame_count})")
        
        if mode == 'alphabet':
            # ALPHABET MODE: Single-frame prediction with hand landmark model
            if alphabet_detector is None:
                result['prediction'] = "Alphabet model not loaded"
                return result
            
            try:
                pred, conf = alphabet_detector.predict_from_frame(img)
                if pred:
                    print(f"[ALPHABET] Detected: {pred} (confidence: {conf:.3f})")
                    # update shared state
                    current_prediction = pred
                    current_confidence = conf
                    result['prediction'] = pred
                    result['confidence'] = float(conf)
                    # Hold-to-commit logic (time based)
                    now = time.monotonic()
                    if pred != last_display_word:
                        last_display_word = pred
                        last_display_start = now
                    shown_for = 0.0 if last_display_start is None else (now - last_display_start)
                    if shown_for >= COMMIT_SECONDS and pred != last_committed_word:
                        recognized_words.append({
                            'text': pred,
                            'confidence': float(conf),
                            't_utc': datetime.utcnow().isoformat() + 'Z'
                        })
                        last_committed_word = pred
                    # Compute sentence
                    committed = [w['text'] for w in recognized_words]
                    sentence = grammar_fix(committed)
                    result['committed_words'] = committed
                    result['sentence'] = sentence
                else:
                    # No hands detected
                    committed = [w['text'] for w in recognized_words]
                    result['committed_words'] = committed
                    result['sentence'] = grammar_fix(committed)
            except Exception as e:
                print(f"Alphabet mode prediction error: {e}")
                committed = [w['text'] for w in recognized_words]
                result['committed_words'] = committed
                result['sentence'] = grammar_fix(committed)
        
        else:
            # MOTION MODE: Multi-frame buffering with CNN-LSTM model
            if detector is None:
                result['prediction'] = "Motion model not loaded"
                return result

            try:
                with detector_lock:
                    # Server-side hand gate: skip buffering/prediction if no hands are present
                    try:
                        hands_present = False
                        try:
                            hands_info = detector.get_hand_overlay_info(img)
                            hands_present = bool(hands_info)
                        except Exception:
                            hands_present = False
                        if not hands_present:
                            # No hands detected; do not add frame to buffer or attempt prediction
                            # Still return current committed sentence if any.
                            committed = [w['text'] for w in recognized_words]
                            result['committed_words'] = committed
                            result['sentence'] = grammar_fix(committed)
                            return result
                    except Exception:
                        # If hand detection fails for any reason, fall back to processing
                        pass

                    detector.add_frame(img)
                    if detector.is_buffer_ready():
                        pred, conf = detector.predict_gesture()
                        if pred:
                            print(f"[MOTION] Detected: {pred} (confidence: {conf:.3f})")
                            # update shared state
                            current_prediction = pred
                            current_confidence = conf
                            result['prediction'] = pred
                            result['confidence'] = float(conf)
                            # Hold-to-commit logic (time based)
                            now = time.monotonic()
                            if pred != last_display_word:
                                last_display_word = pred
                                last_display_start = now
                            shown_for = 0.0 if last_display_start is None else (now - last_display_start)
                            if shown_for >= COMMIT_SECONDS and pred != last_committed_word:
                                recognized_words.append({
                                    'text': pred,
                                    'confidence': float(conf),
                                    't_utc': datetime.utcnow().isoformat() + 'Z'
                                })
                                last_committed_word = pred
                                # Persist session JSON snapshot on each new committed word
                            committed = [w['text'] for w in recognized_words]
                            sentence = grammar_fix(committed)
                            result['committed_words'] = committed
                            result['sentence'] = sentence
                    # Even if no commit, still compute up-to-date sentence for UI (non-invasive)
                    if not result['sentence']:
                        try:
                            committed = [w['text'] for w in recognized_words]
                            result['committed_words'] = committed
                            result['sentence'] = grammar_fix(committed)
                        except Exception:
                            pass
            except Exception as e:
                print(f"Motion mode prediction error: {e}")

        # Draw overlays (kept for internal use / logs) but not returned
        try:
            _ = draw_overlays(img)
        except Exception:
            pass

        # Provide sentence even if no new prediction (reuse existing committed list)
        if not result['sentence']:
            try:
                committed = [w['text'] for w in recognized_words]
                result['committed_words'] = committed
                result['sentence'] = grammar_fix(committed)
            except Exception:
                pass
        
        # Increment frame counter
        frame_count += 1
        
        return result

    except Exception as e:
        print(f"process_frame_bytes_sync error: {e}")
        import traceback as _tb
        _tb.print_exc()
        return result


async def start_ws_server(host='0.0.0.0', port=5001):
        """Start the WebSocket server used for low-latency frame processing.

        Notes:
        - The `websockets.serve` call runs an asyncio-based WebSocket server.
        - We set `max_size` to 4MB to allow reasonably sized JPEG frames from the
            client without rejecting them; adjust if clients send larger images.
        - This coroutine blocks by awaiting an unresolved Future so the server
            keeps running until the process is terminated.
        """
        print(f"Starting WebSocket AI server on ws://{host}:{port}/ai/ws")
        async with websockets.serve(ws_handler, host, port, max_size=4*1024*1024):
                await asyncio.Future()  # run forever


# ============================================================================
# ERROR HANDLERS
# ============================================================================

@app.errorhandler(404)
def not_found(error):
    return jsonify({'status': 'error', 'message': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'status': 'error', 'message': 'Internal server error'}), 500

# ============================================================================
# MAIN APPLICATION
# ============================================================================

if __name__ == '__main__':
    print("SASL Flask AI Backend")
    print("=" * 50)
    print(f"Device: {device}")
    print(f"Classes: {len(class_names) if class_names else 'Not loaded'}")
    print(f"Hand Detection: {'Available' if HAND_DETECTION_AVAILABLE else 'Disabled'}")
    print(f"AI Model: {'Loaded' if detector else 'Failed to load'}")
    print("\nStarting Flask AI server...")
    print("Server will run on: http://localhost:5000")
    print("This server provides AI endpoints for the Node.js frontend")
    print("Press Ctrl+C to stop the server")
    
    try:
        # Start WebSocket server in background thread
        def ws_thread():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(start_ws_server(host='0.0.0.0', port=5001))

        t = threading.Thread(target=ws_thread, daemon=True)
        t.start()

        # Start Flask app (HTTP endpoints)
        app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        if detector:
            detector.cleanup()
        cv2.destroyAllWindows()