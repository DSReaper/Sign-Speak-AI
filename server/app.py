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
from flask import Flask, render_template, Response, jsonify, request
from flask_cors import CORS
import warnings
warnings.filterwarnings("ignore")

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
camera = None
detector = None
current_prediction = "Waiting for camera..."
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
# HAND DETECTION SYSTEM (Same as before)
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
                
                return predicted_class, confidence_score
                
        except Exception as e:
            print(f"CNN prediction error: {e}")
            return None, 0.0
    
    def _predict_with_pose(self):
        """Make prediction using Pose LSTM model"""
        try:
            if self.pose_model is None:
                return None, 0.0
                
            # Prepare pose sequence  
            pose_sequence = np.array(list(self.pose_buffer))
            pose_tensor = torch.FloatTensor(pose_sequence).unsqueeze(0).to(self.device)
            
            # Pose LSTM prediction
            with torch.no_grad():
                outputs = self.pose_model(pose_tensor)
                
                if torch.isnan(outputs).any() or torch.isinf(outputs).any():
                    print("Pose model producing NaN outputs")
                    return None, 0.0
                
                probabilities = torch.softmax(outputs, dim=1)
                confidence, predicted_idx = torch.max(probabilities, 1)
                
                predicted_class = self.class_names[predicted_idx.item()]
                confidence_score = confidence.item()
                
                return predicted_class, confidence_score
                
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
        # Create CNN+LSTM model
        cnn_model = CNNLSTMModel(
            num_classes=num_classes,
            sequence_length=sequence_length,
            input_size=(224, 224)
        ).to(device)
        
        # Create Pose LSTM model
        pose_model = PoseLSTMModel(
            num_classes=num_classes,
            sequence_length=sequence_length,
            pose_dim=225
        ).to(device)
        
        print(f"✓ Model architectures created")
        
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

def generate_frames():
    """Generate video frames for web streaming"""
    global current_prediction, current_confidence, frame_count, camera, is_camera_active
    
    while is_camera_active and camera is not None:
        success, frame = camera.read()
        if not success:
            print("Failed to read from camera")
            break
        
        frame_count += 1
        
        # Flip the frame horizontally for mirror effect
        frame = cv2.flip(frame, 1)
        # Add frame to detector buffer
        if detector:
            detector.add_frame(frame)
            # Make prediction every few frames to reduce computational load
            if frame_count % 3 == 0 and detector.is_buffer_ready():
                prediction, confidence = detector.predict_gesture()
                if prediction:
                    current_prediction = prediction
                    current_confidence = confidence
        # Draw overlays
        frame = draw_overlays(frame)
        # Encode frame as JPEG
        _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        frame_bytes = buffer.tobytes()
        # Yield frame in multipart format
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

def draw_overlays(frame):
    """Draw all overlays on frame"""
    global show_hands
    
    # Hand and pose detection overlay
    if show_hands and detector:
        hands_data = detector.get_hand_overlay_info(frame)
        frame = detector.hand_detector.draw_hands(frame, hands_data)
        
        # Also draw pose landmarks
        frame = detector.hand_detector.draw_pose_landmarks(frame)
    
    # Status overlays
    h, w = frame.shape[:2]
    
    # Buffer status
    if detector and not detector.is_buffer_ready():
        buffer_text = f"Collecting frames: {len(detector.frame_buffer)}/{detector.buffer_size}"
        cv2.putText(frame, buffer_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    
    # Prediction
    if current_prediction not in ["Waiting...", "Waiting for camera..."]:
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
    if detector:
        cv2.putText(frame, "Model: CNN+LSTM + Pose Ensemble", (10, h-70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, "Ensemble AI active", (10, h-45), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    else:
        cv2.putText(frame, "AI Model: Not loaded", (10, h-45), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    
    # Hand detection status
    hand_status = "ON" if show_hands else "OFF"
    cv2.putText(frame, f"Hands: {hand_status}", (w-120, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    return frame

def generate_frames():
    """Generate video frames for web streaming"""
    global current_prediction, current_confidence, frame_count, camera, is_camera_active
    
    while is_camera_active and camera is not None:
        success, frame = camera.read()
        if not success:
            break
        
        frame_count += 1
        
        # Add frame to detector if available
        if detector:
            detector.add_frame(frame)
            
            # Get prediction every few frames
            if frame_count % 3 == 0 and detector.is_buffer_ready():
                prediction, confidence = detector.predict_gesture()
                if prediction:
                    current_prediction = prediction
                    current_confidence = confidence
        
        # Draw overlays
        frame = draw_overlays(frame)
        
        # Encode frame as JPEG
        _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        frame_bytes = buffer.tobytes()
        
        # Yield frame in multipart format
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')


# ============================================================================
# WEB ROUTES (Updated for Node.js integration)
# ============================================================================

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
        
        return jsonify({'status': 'success', 'message': 'AI camera started'})
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
    
    return jsonify({'status': 'success', 'message': 'AI camera stopped'})

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
    
    if detector:
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
        'buffer_ready': detector.is_buffer_ready() if detector else False,
        'buffer_size': len(detector.frame_buffer) if detector else 0,
        'is_camera_active': is_camera_active,
        'model_loaded': detector is not None
    })

@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'ai_model': 'loaded' if detector else 'not_loaded',
        'camera': 'active' if is_camera_active else 'inactive',
        'device': str(device)
    })

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
        app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        if camera:
            camera.release()
        if detector:
            detector.cleanup()
        cv2.destroyAllWindows()