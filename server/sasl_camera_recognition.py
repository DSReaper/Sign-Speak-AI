#!/usr/bin/env python3
"""
SASL PyTorch Camera Recognition System
====================================

Real-time SASL sign recognition using PyTorch models trained with the video-based system.
Provides live camera feed with sign recognition, confidence scores, and visual feedback.

Features:
- Real-time video processing with PyTorch models
- CNN+LSTM and Pose LSTM ensemble predictions
- MediaPipe pose/hand landmark overlay
- Confidence-based prediction filtering
- Top-3 predictions display
- Clean, professional interface
"""

import torch
import torch.nn as nn
import cv2
import numpy as np
import json
import mediapipe as mp
from pathlib import Path
import time
from collections import deque
import timm

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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

class SASLCameraRecognition:
    """
    Real-time SASL recognition using PyTorch models
    """
    
    def __init__(self, cnn_model_path, pose_model_path, classes_path, 
                 sequence_length=30, input_size=(224, 224), confidence_threshold=0.3):
        """
        Initialize the PyTorch-based camera recognition system
        
        Args:
            cnn_model_path: Path to CNN+LSTM PyTorch model (.pth)
            pose_model_path: Path to Pose LSTM PyTorch model (.pth) 
            classes_path: Path to class names JSON file
            sequence_length: Number of frames for temporal modeling
            input_size: Input image size for CNN
            confidence_threshold: Minimum confidence for predictions
        """
        self.sequence_length = sequence_length
        self.input_size = input_size
        self.confidence_threshold = confidence_threshold
        
        # Load class names
        with open(classes_path, 'r') as f:
            self.class_names = json.load(f)
        self.num_classes = len(self.class_names)
        
        print(f"Initializing PyTorch SASL Camera Recognition")
        print(f"Device: {device}")
        print(f"Classes: {self.num_classes}")
        print(f"Sequence length: {sequence_length}")
        print(f"Input size: {input_size}")
        
        # Load PyTorch models
        print("Loading PyTorch models...")
        
        # CNN+LSTM Model
        self.cnn_model = CNNLSTMModel(self.num_classes, sequence_length, input_size)
        self.cnn_model.load_state_dict(torch.load(cnn_model_path, map_location=device))
        self.cnn_model.to(device)
        self.cnn_model.eval()
        
        # Pose LSTM Model  
        self.pose_model = PoseLSTMModel(self.num_classes, sequence_length)
        self.pose_model.load_state_dict(torch.load(pose_model_path, map_location=device))
        self.pose_model.to(device)
        self.pose_model.eval()
        
        print("PyTorch models loaded successfully")
        
        # Initialize MediaPipe
        print("Initializing MediaPipe...")
        self.mp_pose = mp.solutions.pose
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        self.pose_detector = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        self.hand_detector = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            model_complexity=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Frame buffers for temporal modeling
        self.frame_buffer = deque(maxlen=sequence_length)
        self.pose_buffer = deque(maxlen=sequence_length)
        
        # Prediction smoothing
        self.prediction_buffer = deque(maxlen=5)  # Smooth over 5 predictions
        
        print("Camera recognition system ready!")
    
    def extract_pose_landmarks(self, rgb_frame):
        """Extract pose and hand landmarks from frame"""
        landmarks = []
        
        # Process pose and hands
        pose_results = self.pose_detector.process(rgb_frame)
        hand_results = self.hand_detector.process(rgb_frame)
        
        # Add pose landmarks (33 points × 3 coordinates = 99 features)
        if pose_results.pose_landmarks:
            for landmark in pose_results.pose_landmarks.landmark:
                landmarks.extend([landmark.x, landmark.y, landmark.z])
        else:
            landmarks.extend([0.0] * 99)
        
        # Add hand landmarks (2 hands × 21 points × 3 coordinates = 126 features)
        hands_added = 0
        if hand_results.multi_hand_landmarks:
            for hand_landmarks in hand_results.multi_hand_landmarks:
                if hands_added < 2:
                    for landmark in hand_landmarks.landmark:
                        landmarks.extend([landmark.x, landmark.y, landmark.z])
                    hands_added += 1
        
        # Pad with zeros if less than 2 hands detected
        while hands_added < 2:
            landmarks.extend([0.0] * 63)  # 21 points × 3 coordinates
            hands_added += 1
        
        return landmarks[:225], pose_results, hand_results  # Ensure consistent size
    
    def predict_sign(self):
        """Make prediction using both models"""
        if len(self.frame_buffer) < self.sequence_length:
            return None, None, []
        
        # Prepare video sequence for CNN+LSTM
        video_sequence = np.array(list(self.frame_buffer)) / 255.0
        video_tensor = torch.FloatTensor(video_sequence).unsqueeze(0).permute(0, 1, 4, 2, 3).to(device)
        
        # Prepare pose sequence for Pose LSTM
        pose_sequence = np.array(list(self.pose_buffer))
        pose_tensor = torch.FloatTensor(pose_sequence).unsqueeze(0).to(device)
        
        with torch.no_grad():
            # CNN+LSTM prediction
            cnn_outputs = self.cnn_model(video_tensor)
            cnn_probs = torch.softmax(cnn_outputs, dim=1)
            
            # Pose LSTM prediction
            pose_outputs = self.pose_model(pose_tensor)
            pose_probs = torch.softmax(pose_outputs, dim=1)
            
            # Ensemble prediction (average probabilities)
            ensemble_probs = (cnn_probs + pose_probs) / 2
            
            # Get top-3 predictions
            top3_probs, top3_indices = torch.topk(ensemble_probs, 3, dim=1)
            
            top3_predictions = []
            for i in range(3):
                class_idx = top3_indices[0, i].item()
                confidence = top3_probs[0, i].item()
                class_name = self.class_names[class_idx]
                top3_predictions.append((class_name, confidence))
            
            # Best prediction
            best_class = top3_predictions[0][0]
            best_confidence = top3_predictions[0][1]
            
            return best_class, best_confidence, top3_predictions
    
    def smooth_prediction(self, prediction, confidence):
        """Smooth predictions over time to reduce flicker"""
        self.prediction_buffer.append((prediction, confidence))
        
        if len(self.prediction_buffer) < 3:
            return prediction, confidence
        
        # Count occurrences of each prediction
        prediction_counts = {}
        total_confidence = 0
        
        for pred, conf in self.prediction_buffer:
            if pred not in prediction_counts:
                prediction_counts[pred] = []
            prediction_counts[pred].append(conf)
            total_confidence += conf
        
        # Find most frequent prediction with highest average confidence
        best_pred = None
        best_score = 0
        
        for pred, confidences in prediction_counts.items():
            avg_confidence = sum(confidences) / len(confidences)
            frequency_score = len(confidences) / len(self.prediction_buffer)
            combined_score = avg_confidence * frequency_score
            
            if combined_score > best_score:
                best_score = combined_score
                best_pred = pred
        
        return best_pred, best_score
    
    def draw_landmarks(self, frame, pose_results, hand_results):
        """Draw MediaPipe landmarks on frame"""
        # Draw pose landmarks
        if pose_results.pose_landmarks:
            self.mp_drawing.draw_landmarks(
                frame,
                pose_results.pose_landmarks,
                self.mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style()
            )
        
        # Draw hand landmarks
        if hand_results.multi_hand_landmarks:
            for hand_landmarks in hand_results.multi_hand_landmarks:
                self.mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS,
                    self.mp_drawing_styles.get_default_hand_landmarks_style(),
                    self.mp_drawing_styles.get_default_hand_connections_style()
                )
    
    def draw_predictions(self, frame, best_prediction, best_confidence, top3_predictions):
        """Draw prediction results on frame"""
        height, width = frame.shape[:2]
        
        # Main prediction
        if best_prediction and best_confidence > self.confidence_threshold:
            # Background for main prediction
            cv2.rectangle(frame, (10, 10), (width - 10, 80), (0, 0, 0), -1)
            cv2.rectangle(frame, (10, 10), (width - 10, 80), (0, 255, 0), 2)
            
            # Main prediction text
            main_text = f"Sign: {best_prediction}"
            conf_text = f"Confidence: {best_confidence:.1%}"
            
            cv2.putText(frame, main_text, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(frame, conf_text, (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        else:
            # No confident prediction
            cv2.rectangle(frame, (10, 10), (width - 10, 50), (0, 0, 0), -1)
            cv2.rectangle(frame, (10, 10), (width - 10, 50), (0, 0, 255), 2)
            cv2.putText(frame, "No sign detected", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Top-3 predictions
        if top3_predictions:
            y_start = 100
            cv2.rectangle(frame, (10, y_start), (350, y_start + 120), (0, 0, 0), -1)
            cv2.rectangle(frame, (10, y_start), (350, y_start + 120), (255, 255, 0), 2)
            
            cv2.putText(frame, "Top 3 Predictions:", (20, y_start + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            for i, (pred, conf) in enumerate(top3_predictions):
                y_pos = y_start + 45 + i * 25
                text = f"{i+1}. {pred}: {conf:.1%}"
                color = (0, 255, 0) if i == 0 else (255, 255, 255)
                cv2.putText(frame, text, (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        # Instructions
        instructions = [
            "Instructions:",
            "- Hold sign clearly for 1-2 seconds",
            "- Ensure good lighting",
            "- Press 'q' to quit, 'r' to reset"
        ]
        
        y_start = height - 120
        cv2.rectangle(frame, (10, y_start), (400, height - 10), (0, 0, 0), -1)
        cv2.rectangle(frame, (10, y_start), (400, height - 10), (0, 255, 255), 2)
        
        for i, instruction in enumerate(instructions):
            y_pos = y_start + 20 + i * 20
            cv2.putText(frame, instruction, (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    
    def run_live_recognition(self):
        """Run live camera recognition"""
        print("Starting live SASL recognition...")
        print("Controls:")
        print("  - Hold signs clearly for 1-2 seconds")
        print("  - Press 'q' to quit")
        print("  - Press 'r' to reset prediction buffer")
        
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("ERROR: Could not open camera")
            return
        
        # Set camera properties
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        fps_counter = 0
        fps_start_time = time.time()
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("ERROR: Could not read frame")
                    break
                
                # Flip frame horizontally for mirror effect
                frame = cv2.flip(frame, 1)
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Resize frame for model input
                model_frame = cv2.resize(rgb_frame, self.input_size)
                
                # Extract landmarks and update buffers
                landmarks, pose_results, hand_results = self.extract_pose_landmarks(rgb_frame)
                
                self.frame_buffer.append(model_frame)
                self.pose_buffer.append(landmarks)
                
                # Make prediction
                best_prediction, best_confidence, top3_predictions = self.predict_sign()
                
                # Smooth prediction
                if best_prediction:
                    best_prediction, best_confidence = self.smooth_prediction(best_prediction, best_confidence)
                
                # Draw landmarks
                self.draw_landmarks(frame, pose_results, hand_results)
                
                # Draw predictions
                self.draw_predictions(frame, best_prediction, best_confidence, top3_predictions)
                
                # Calculate and display FPS
                fps_counter += 1
                if fps_counter % 30 == 0:
                    fps_end_time = time.time()
                    fps = 30 / (fps_end_time - fps_start_time)
                    fps_start_time = fps_end_time
                
                # FPS display
                cv2.putText(frame, f"FPS: {fps:.1f}" if 'fps' in locals() else "FPS: --", 
                          (frame.shape[1] - 100, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                # Display frame
                cv2.imshow('SASL PyTorch Recognition', frame)
                
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('r'):
                    # Reset buffers
                    self.frame_buffer.clear()
                    self.pose_buffer.clear()
                    self.prediction_buffer.clear()
                    print("Buffers reset")
        
        except KeyboardInterrupt:
            print("\nRecognition stopped by user")
        
        finally:
            cap.release()
            cv2.destroyAllWindows()
            self.pose_detector.close()
            self.hand_detector.close()
            print("Camera recognition finished")

def main():
    """Main function for testing"""
    print("Searching for trained PyTorch models...")
    
    # Check for the latest training outputs
    outputs_dir = Path("outputs")
    model_files = []
    
    if outputs_dir.exists():
        # Find the latest training directory
        training_dirs = [d for d in outputs_dir.iterdir() if d.is_dir() and d.name.startswith("training_")]
        if training_dirs:
            # Sort by name (timestamp) and get the latest
            latest_training_dir = sorted(training_dirs, key=lambda x: x.name)[-1]
            print(f"Found latest training session: {latest_training_dir.name}")
            
            # Check for models in the latest training directory
            models_dir = latest_training_dir / "models"
            results_dir = latest_training_dir / "results"
            
            if models_dir.exists() and results_dir.exists():
                cnn_model_path = models_dir / "best_sasl_cnn_lstm_model.pth"
                pose_model_path = models_dir / "best_sasl_pose_lstm_model.pth"
                
                # Check for class names - try both pytorch_sasl_classes.json and class_names.json
                classes_path = results_dir / "pytorch_sasl_classes.json"
                if not classes_path.exists():
                    classes_path = results_dir / "class_names.json"
                
                if cnn_model_path.exists() and pose_model_path.exists() and classes_path.exists():
                    model_files = [str(cnn_model_path), str(pose_model_path), str(classes_path)]
                    print(f"Found CNN+LSTM model: {cnn_model_path}")
                    print(f"Found Pose LSTM model: {pose_model_path}")
                    print(f"Found class names: {classes_path}")
                else:
                    print(f"Missing model files in {models_dir}")
            else:
                print(f"Models or results directory not found in {latest_training_dir}")
        else:
            print("No training directories found in outputs/")
    
    # Fallback: Check for models in root directory or outputs/
    if not model_files:
        print("Checking for models in root directory...")
        fallback_files = [
            "best_sasl_cnn_lstm_model.pth",
            "best_sasl_pose_lstm_model.pth", 
            "pytorch_sasl_classes.json"
        ]
        
        # Check outputs directory first, then root
        for model_file in fallback_files:
            if (outputs_dir / model_file).exists():
                model_files.append(str(outputs_dir / model_file))
            elif Path(model_file).exists():
                model_files.append(model_file)
            else:
                model_files.append(None)
        
        # Check if all files found
        if None in model_files:
            missing_files = [f for f, path in zip(fallback_files, model_files) if path is None]
            print(f"ERROR: Missing PyTorch model files: {missing_files}")
            print("\nTo fix this issue:")
            print("1. Run training first: python video_based_sasl_training.py")
            print("2. Or collect training data: python sasl_video_collector.py")
            print("3. Make sure training completes successfully")
            return
        else:
            print("Found models in fallback locations")
    
    if not model_files or len(model_files) != 3:
        print("ERROR: Could not locate all required model files")
        print("\nTo fix this issue:")
        print("1. Run training first: python video_based_sasl_training.py") 
        print("2. Or collect training data: python sasl_video_collector.py")
        print("3. Make sure training completes successfully")
        return
    
    try:
        print("\nStarting SASL Camera Recognition System...")
        print("Initializing camera and loading models...")
        
        # Initialize and run recognition
        recognition = SASLCameraRecognition(
            cnn_model_path=model_files[0],
            pose_model_path=model_files[1],
            classes_path=model_files[2]
        )
        
        print("Models loaded successfully!")
        print("Starting live recognition...")
        print("\nControls:")
        print("   Q = Quit")
        print("   R = Reset prediction buffer")
        print("   U = Toggle UI overlay")
        print("   M = Toggle MediaPipe landmarks")
        print("\nCamera window will open shortly...")
        
        recognition.run_live_recognition()
        
    except FileNotFoundError as e:
        print(f"ERROR: Model file not found: {e}")
        print("Please check that all model files exist and are accessible")
    except Exception as e:
        print(f"ERROR during recognition: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()