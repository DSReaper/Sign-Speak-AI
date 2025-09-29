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

# Suppress MediaPipe verbose logging
os.environ['GLOG_minloglevel'] = '2'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['MEDIAPIPE_DISABLE_GPU'] = '1'

try:
    import mediapipe as mp
    import timm
    HAND_DETECTION_AVAILABLE = True
except ImportError:
    print("MediaPipe or timm not available. Hand detection disabled.")
    HAND_DETECTION_AVAILABLE = False

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for Node.js integration

# Global variables
detector = None
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

# Import grammar utilities (rule-based grammar + optional session persistence)
# We try relative import first (works when `server` is a package). Fallback to
# plain import so that running `python server/app.py` still functions.
try:  # packaged execution
    from .grammar_utils import grammar_fix, save_session_json  # type: ignore
except Exception:
    try:  # script-style execution
        from grammar_utils import grammar_fix, save_session_json  # type: ignore
    except Exception:
        # Minimal fallbacks so the server never hard-crashes just because the
        # grammar helper is missing. This degrades functionality only.
        def grammar_fix(words):  # type: ignore
            return " ".join(words).strip()
        def save_session_json(out_dir, words, sentence, meta):  # type: ignore
            return ""  # no-op
        print("WARNING: grammar_utils import failed; using fallback grammar + no-op save_session_json")

# Feature flags / lightweight configuration
ENABLE_SESSION_JSON = True  # persist sessions when a new word is committed
SESSION_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sessions")
_last_session_path = None  # updated on each save

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
        if not HAND_DETECTION_AVAILABLE:
            raise ImportError("timm is required for CNNLSTMModel")
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
        self.lstm1 = nn.LSTM(512, 256, bidirectional=True, batch_first=True, dropout=0.3)
        self.lstm2 = nn.LSTM(512, 128, bidirectional=True, batch_first=True, dropout=0.3)
        
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
        x, _ = self.lstm2(x)  # (batch, seq, 256)
        
        # Global average pooling over sequence
        x = torch.mean(x, dim=1)  # (batch, 256)
        
        # Classification
        x = self.classifier(x)
        
        return x

class PoseLSTMModel(nn.Module):
    """LSTM model for pose sequence classification using PyTorch"""
    
    def __init__(self, num_classes, sequence_length=30, pose_dim=225):
        super(PoseLSTMModel, self).__init__()
        
        self.sequence_length = sequence_length
        self.pose_dim = pose_dim
        self.num_classes = num_classes
        
        # Input processing
        self.input_bn = nn.BatchNorm1d(pose_dim)
        self.input_dropout = nn.Dropout(0.2)
        
        # LSTM layers
        self.lstm1 = nn.LSTM(pose_dim, 256, bidirectional=True, batch_first=True, dropout=0.4)
        self.lstm2 = nn.LSTM(512, 128, bidirectional=True, batch_first=True, dropout=0.3)
        self.lstm3 = nn.LSTM(256, 64, bidirectional=True, batch_first=True, dropout=0.3)
        
        # Classification layers
        self.classifier = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        batch_size, seq_len, pose_dim = x.size()
        
        # Normalize input
        x = x.view(-1, pose_dim)  # (batch*seq, pose_dim)
        x = self.input_bn(x)
        x = self.input_dropout(x)
        x = x.view(batch_size, seq_len, pose_dim)  # (batch, seq, pose_dim)
        
        # LSTM layers
        x, _ = self.lstm1(x)  # (batch, seq, 512)
        x, _ = self.lstm2(x)  # (batch, seq, 256)
        x, _ = self.lstm3(x)  # (batch, seq, 128)
        
        # Global average pooling over sequence
        x = torch.mean(x, dim=1)  # (batch, 128)
        
        # Classification
        x = self.classifier(x)
        
        return x

class HandFocusedCNN_LSTM(nn.Module):
    """Legacy model class for backward compatibility"""
    
    def __init__(self, cnn, hidden_size=256, num_classes=41, num_layers=2, dropout=0.3):
        super(HandFocusedCNN_LSTM, self).__init__()
        print("Warning: Using legacy HandFocusedCNN_LSTM. Consider using CNNLSTMModel for better performance.")
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
    """MediaPipe-based hand and pose detection for web application"""
    
    def __init__(self):
        if not HAND_DETECTION_AVAILABLE:
            self.hands = None
            self.pose = None
            return
            
        self.mp_hands = mp.solutions.hands
        self.mp_pose = mp.solutions.pose
        self.mp_draw = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.3
        )
        
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
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
    
    def extract_pose_landmarks(self, frame):
        """Extract pose and hand landmarks from frame for model input"""
        if not self.hands or not self.pose:
            return np.zeros(225)  # Default pose dimension
            
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        landmarks = []
        
        # Process pose
        pose_results = self.pose.process(rgb_frame)
        # Process hands
        hand_results = self.hands.process(rgb_frame)
        
        # Add pose landmarks (33 points × 3 coordinates = 99 features)
        if pose_results.pose_landmarks:
            for landmark in pose_results.pose_landmarks.landmark:
                landmarks.extend([landmark.x, landmark.y, landmark.z])
        else:
            landmarks.extend([0.0] * 99)
        
        # Add hand landmarks (2 hands × 21 points × 3 coordinates = 126 features)
        hands_added = 0
        if hand_results.multi_hand_landmarks:
            for hand_landmarks in hand_results.multi_hand_landmarks[:2]:  # Max 2 hands
                for landmark in hand_landmarks.landmark:
                    landmarks.extend([landmark.x, landmark.y, landmark.z])
                hands_added += 1
        
        # Pad with zeros if less than 2 hands detected
        while hands_added < 2:
            landmarks.extend([0.0] * 63)  # 21 points × 3 coordinates
            hands_added += 1
        
        return np.array(landmarks)  # Total: 99 + 126 = 225 features
    
    def draw_hands(self, frame, hands_data):
        """Draw hand landmarks and bounding boxes"""
        if not self.hands:
            return frame
        # Define colors for each finger (thumb, index, middle, ring, pinky)
        finger_colors = [
            (255, 0, 0),    # Thumb - Blue
            (0, 255, 0),    # Index - Green
            (0, 0, 255),    # Middle - Red
            (255, 255, 0),  # Ring - Cyan
            (255, 0, 255)   # Pinky - Magenta
        ]
        # Finger landmark indices in MediaPipe
        finger_indices = [
            [0, 1, 2, 3, 4],      # Thumb
            [0, 5, 6, 7, 8],      # Index
            [0, 9, 10, 11, 12],   # Middle
            [0, 13, 14, 15, 16],  # Ring
            [0, 17, 18, 19, 20]   # Pinky
        ]
        for hand_data in hands_data:
            hand_landmarks = hand_data['landmarks']
            h, w, _ = frame.shape
            # Draw colored fingers
            for f, indices in enumerate(finger_indices):
                for i in range(len(indices)-1):
                    start = hand_landmarks.landmark[indices[i]]
                    end = hand_landmarks.landmark[indices[i+1]]
                    x1, y1 = int(start.x * w), int(start.y * h)
                    x2, y2 = int(end.x * w), int(end.y * h)
                    cv2.line(frame, (x1, y1), (x2, y2), finger_colors[f], 3)
            # Draw all landmarks as small circles
            for idx, lm in enumerate(hand_landmarks.landmark):
                x, y = int(lm.x * w), int(lm.y * h)
                cv2.circle(frame, (x, y), 4, (255, 255, 255), -1)
            # Draw bounding box
            bbox = hand_data['bbox']
            cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
        return frame
    
    def draw_pose_landmarks(self, frame):
        """Draw pose landmarks on frame"""
        if not self.pose:
            return frame
            
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pose_results = self.pose.process(rgb_frame)
        
        if pose_results.pose_landmarks:
            self.mp_draw.draw_landmarks(
                frame, 
                pose_results.pose_landmarks, 
                self.mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style()
            )
        
        return frame
    
    def close(self):
        """Clean up resources"""
        if hasattr(self, 'hands') and self.hands:
            self.hands.close()
        if hasattr(self, 'pose') and self.pose:
            self.pose.close()

# ============================================================================
# GESTURE DETECTION SYSTEM
# ============================================================================

class WebGestureDetector:
    """Web-based ensemble gesture detection system using CNN+LSTM and Pose LSTM"""
    
    def __init__(self, cnn_model, device, class_names, buffer_size=16):
        self.cnn_model = cnn_model
        self.pose_model = pose_model_global  # Get pose model from global
        self.device = device
        self.class_names = class_names
        self.buffer_size = buffer_size
        self.frame_buffer = deque(maxlen=buffer_size)
        self.pose_buffer = deque(maxlen=buffer_size) 
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
        """Add frame to buffer and extract pose landmarks"""
        self.frame_buffer.append(frame.copy())
        
        # Extract pose landmarks for this frame
        pose_landmarks = self.hand_detector.extract_pose_landmarks(frame)
        self.pose_buffer.append(pose_landmarks)
    
    def is_buffer_ready(self):
        """Check if buffer has enough frames for prediction"""
        return len(self.frame_buffer) >= self.buffer_size and len(self.pose_buffer) >= self.buffer_size
    
    def predict_gesture(self):
        """Predict gesture using ensemble of CNN+LSTM and Pose LSTM models"""
        if not self.is_buffer_ready():
            return None, 0.0
        
        try:
            cnn_prediction, cnn_confidence = self._predict_with_cnn()
            pose_prediction, pose_confidence = self._predict_with_pose()
            
            # Ensemble prediction - weighted average
            ensemble_prediction, ensemble_confidence = self._ensemble_predictions(
                (cnn_prediction, cnn_confidence),
                (pose_prediction, pose_confidence)
            )
            
            if ensemble_prediction is None:
                return None, 0.0
                
            # Filter low confidence predictions
            if ensemble_confidence < 0.01:
                return None, 0.0
            
            # Add to history
            self.prediction_history.append((ensemble_prediction, ensemble_confidence))
            
            # Get stable prediction
            stable_prediction = self._get_stable_prediction()
            return stable_prediction, ensemble_confidence
            
        except Exception as e:
            print(f"Prediction error: {e}")
            return None, 0.0
    
    def _predict_with_cnn(self):
        """Make prediction using CNN+LSTM model"""
        try:
            # Preprocess frames
            frames_tensor = self._preprocess_frames(list(self.frame_buffer))
            frames_tensor = frames_tensor.unsqueeze(0).to(self.device)
            
            # CNN+LSTM prediction
            with torch.no_grad():
                outputs = self.cnn_model(frames_tensor)
                
                if torch.isnan(outputs).any() or torch.isinf(outputs).any():
                    print("CNN model producing NaN outputs")
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
            print(f"Pose prediction error: {e}")
            return None, 0.0
    
    def _ensemble_predictions(self, cnn_result, pose_result):
        """Combine CNN and Pose predictions with weighted ensemble"""
        cnn_pred, cnn_conf = cnn_result
        pose_pred, pose_conf = pose_result
        
        # If either model failed, use the other
        if cnn_pred is None and pose_pred is None:
            return None, 0.0
        elif cnn_pred is None:
            return pose_pred, pose_conf * 0.8  # Reduced confidence for single model
        elif pose_pred is None:
            return cnn_pred, cnn_conf * 0.8
        
        # Both models made predictions
        # Weight: CNN 60%, Pose 40% (CNN generally more accurate for gesture recognition)
        cnn_weight = 0.6
        pose_weight = 0.4
        
        # If both models predict the same class, increase confidence
        if cnn_pred == pose_pred:
            ensemble_confidence = cnn_weight * cnn_conf + pose_weight * pose_conf
            return cnn_pred, min(ensemble_confidence * 1.2, 1.0)  # Boost confidence but cap at 1.0
        else:
            # Different predictions - choose the more confident one but reduce overall confidence
            if cnn_conf >= pose_conf:
                return cnn_pred, cnn_conf * 0.7  # Reduce confidence for disagreement
            else:
                return pose_pred, pose_conf * 0.7
    
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

def validate_models(cnn_model, pose_model, device, class_names, sequence_length=16):
    """Validate that both models produce valid outputs"""
    try:
        cnn_model.eval()
        pose_model.eval()
        
        # Test CNN+LSTM model
        dummy_video = torch.randn(1, sequence_length, 3, 224, 224).to(device)
        with torch.no_grad():
            cnn_output = cnn_model(dummy_video)
            
            # Check CNN output shape
            if cnn_output.shape != (1, len(class_names)):
                print(f"CNN model output shape mismatch: expected (1, {len(class_names)}), got {cnn_output.shape}")
                return False
            
            # Check for NaN or infinite values
            if torch.isnan(cnn_output).any() or torch.isinf(cnn_output).any():
                print("CNN model produces NaN or infinite values")
                return False
        
        # Test Pose LSTM model  
        dummy_pose = torch.randn(1, sequence_length, 225).to(device)  # 225 features
        with torch.no_grad():
            pose_output = pose_model(dummy_pose)
            
            # Check Pose output shape
            if pose_output.shape != (1, len(class_names)):
                print(f"Pose model output shape mismatch: expected (1, {len(class_names)}), got {pose_output.shape}")
                return False
            
            # Check for NaN or infinite values
            if torch.isnan(pose_output).any() or torch.isinf(pose_output).any():
                print("Pose model produces NaN or infinite values")
                return False
        
        print("✓ Both models validation passed")
        return True
        
    except Exception as e:
        print(f"Models validation failed: {e}")
        return False

def load_models_and_classes():
    """Load both CNN+LSTM and Pose LSTM models with class names"""
    # Load class names
    class_names_path = os.path.join(current_dir, "models", "class_names.json")
    try:
        with open(class_names_path, 'r') as f:
            class_names = json.load(f)
        print(f"✓ Loaded {len(class_names)} classes")
    except Exception as e:
        print(f"Error loading class names: {e}")
        return None, None, None
    
    num_classes = len(class_names)
    sequence_length = 16  # Reduced from 30 for faster processing
    
    # Load CNN+LSTM model
    cnn_model_path = os.path.join(current_dir, "models", "best_sasl_cnn_lstm_model.pth")
    pose_model_path = os.path.join(current_dir, "models", "best_sasl_pose_lstm_model.pth")
    
    if not os.path.exists(cnn_model_path):
        print(f"CNN model file not found: {cnn_model_path}")
        return None, None, None
    
    if not os.path.exists(pose_model_path):
        print(f"Pose model file not found: {pose_model_path}")
        return None, None, None
    
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
        
        # Load CNN+LSTM weights
        cnn_checkpoint = torch.load(cnn_model_path, map_location=device, weights_only=False)
        cnn_model.load_state_dict(cnn_checkpoint, strict=False)
        cnn_model.eval()
        print(f"✓ CNN+LSTM model weights loaded successfully")
        
        # Load Pose LSTM weights  
        pose_checkpoint = torch.load(pose_model_path, map_location=device, weights_only=False)
        pose_model.load_state_dict(pose_checkpoint, strict=False)
        pose_model.eval()
        print(f"✓ Pose LSTM model weights loaded successfully")
        
        # Validate both models
        if not validate_models(cnn_model, pose_model, device, class_names, sequence_length):
            print(f"Models validation failed")
            return None, None, None
        
        return cnn_model, pose_model, class_names
        
    except Exception as e:
        print(f"Error loading models: {e}")
        return None, None, None

# Legacy function for backward compatibility
def load_model_and_classes():
    """Legacy function - loads both models but returns only CNN model for compatibility"""
    cnn_model, pose_model, class_names = load_models_and_classes()
    if cnn_model is None:
        return None, None
    # Store pose model globally for access by detector
    global pose_model_global
    pose_model_global = pose_model
    return cnn_model, class_names

# Initialize global pose model variable
pose_model_global = None

# Initialize model and detector
print("Loading SASL AI Model...")
model, class_names = load_model_and_classes()
if model is None or class_names is None:
    print("Failed to load model or class names. AI features will be disabled.")
    detector = None
else:
    detector = WebGestureDetector(model, device, class_names)

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
        'show_hands': show_hands,
        'show_server_overlays': show_server_overlays,
        'buffer_ready': detector.is_buffer_ready() if detector else False,
        'buffer_size': len(detector.frame_buffer) if detector else 0,
        'model_loaded': detector is not None,
        'committed_words': committed,
        'sentence': sentence,
        'session_json_path': _last_session_path
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
    """WebSocket handler for low-latency frame processing.

    Behavior and message flow:
    - Expects binary messages containing JPEG bytes from the client.
    - For binary messages: decodes -> process (detector/prediction/overlays) -> encodes -> send back binary JPEG bytes.
    - For text messages: replies with a simple acknowledgement ('OK').

    Implementation notes:
    - cv2 and torch are blocking; we offload the heavy work to a threadpool via
    `asyncio.get_event_loop().run_in_executor(...)` which calls the sync wrapper
    `process_frame_bytes_sync`. That wrapper performs locking around the shared
    detector object and runs the blocking code safely.
    - This handler is defensive: it logs sizes/previews instead of dumping raw
    binary payloads, catches exceptions, and attempts best-effort notifications
    back to the client when errors occur.
    """
    # Logging for debugging connection issues
    # We'll create a per-connection background worker with a bounded queue (size 1)
    # that always keeps the latest frame and drops older ones. The worker runs
    # `process_frame_bytes_sync` in its own thread and sends the processed bytes
    # back to the websocket using the connection's asyncio loop. This prevents
    # blocking the asyncio event loop on CPU-bound CV/model work and avoids
    # unbounded memory growth when clients send frames faster than processing.
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
                    # Text messages (log content up to 200 chars)
                    text_preview = str(message)[:200]
                    print(f"WS: received text message: {text_preview}")
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

    Returns:
    - dict with keys: 'prediction' (str or None), 'confidence' (float)
    """
  # The json object that will be sent back to the client
    result = {
        'prediction': None,
        'confidence': 0.0,
        'committed_words': [],
        'sentence': "",
    }

    try:
        # Decode bytes and log basic diagnostics
        nparr = np.frombuffer(frame_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            print("process_frame_bytes_sync: cv2.imdecode failed for incoming frame")
            return result

        # optional debug metric
        try:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            mean_brightness = float(np.mean(gray))
        except Exception:
            mean_brightness = -1.0

        # Interact with the detector under a lock to ensure thread-safety
        if detector is None:
            return result

        try:
            with detector_lock:
                detector.add_frame(img)
                if detector.is_buffer_ready():
                    pred, conf = detector.predict_gesture()
                    if pred:
                        # update shared state
                        global current_prediction, current_confidence
                        global last_display_word, last_display_start, last_committed_word
                        global recognized_words
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
                            if ENABLE_SESSION_JSON:
                                try:
                                    committed_words_snapshot = [w for w in recognized_words]
                                    sentence_snapshot = grammar_fix([w['text'] for w in committed_words_snapshot])
                                    meta = {"model_type": "Hand-Focused SASL"}
                                    global _last_session_path
                                    _last_session_path = save_session_json(SESSION_DIR, committed_words_snapshot, sentence_snapshot, meta)
                                except Exception as save_exc:
                                    # Non-fatal: just log once per failure kind (simple print here)
                                    print(f"Session save failed: {save_exc}")
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
        except Exception as det_e:
            print(f"Detector processing error: {det_e}")
            import traceback as _tb
            _tb.print_exc()

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
        return result

    except Exception as e:
        print(f"process_frame_bytes_sync error: {e}")
        import traceback as _tb
        _tb.print_exc()
        return result


async def start_ws_server(host='0.0.0.0', port=5001):
        """Start the WebSocket server used for low-latency AI frame processing.

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
    print("🤟 SASL Flask AI Backend 🤟")
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