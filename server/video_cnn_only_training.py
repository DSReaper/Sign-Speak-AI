#!/usr/bin/env python3
"""
SASL CNN-Only Video Training System
==================================

Streamlined version that trains only the CNN+LSTM model for better performance.
Based on analysis showing CNN+LSTM achieves 90-100% accuracy vs lower pose accuracy.

Key Features:
- CNN+LSTM architecture with EfficientNet backbone
- No pose dependency - more robust and faster
- Simplified training pipeline
- Better performance through focused approach
"""

import torch
import torch.nn as nn
import torch.optim as optim
import cv2
import numpy as np
import json
import pickle
import timm
import random
import time
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import multiprocessing as mp_cpu
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader

# MediaPipe for hand detection
try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
    print("MediaPipe available for hand detection")
except ImportError:
    MEDIAPIPE_AVAILABLE = False
    print("Warning: MediaPipe not available. Install with: pip install mediapipe")
    print("Hand detection will be disabled.")

# Set device and configure PyTorch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
else:
    print("Using CPU - Consider GPU for faster training")

# Set seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)

# ImageNet normalization constants (match EfficientNet pretraining)
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406])
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225])

def process_single_video(args):
    """
    Process a single video file for CNN + hand landmark extraction
    """
    video_path, sequence_length, input_size, cache_dir = args
    
    try:
        # Check cache first
        cache_filename = f"{Path(video_path).stem}_{sequence_length}_{input_size[0]}x{input_size[1]}_cnn_hands.pkl"
        cache_path = cache_dir / cache_filename
        
        if cache_path.exists():
            try:
                with open(cache_path, 'rb') as f:
                    cached_data = pickle.load(f)
                    if (cached_data.get('sequence_length') == sequence_length and
                        cached_data.get('input_size') == input_size):
                        return (video_path, cached_data['video_seq'], cached_data['hand_landmarks'], True)
            except:
                pass  # Cache corrupted, process normally
        
        # Initialize MediaPipe Hand detection
        hand_landmarks_seq = []
        if MEDIAPIPE_AVAILABLE:
            mp_hands = mp.solutions.hands
            hands = mp_hands.Hands(
                static_image_mode=False,
                max_num_hands=2,  # Detect both hands
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
        
        # Process video
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return (video_path, None, None, False)
        
        frames = []
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Resize frame for CNN
            frame_resized = cv2.resize(frame, input_size)
            rgb_frame = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
            frames.append(rgb_frame)
            
            # Extract hand landmarks using MediaPipe
            if MEDIAPIPE_AVAILABLE:
                results = hands.process(rgb_frame)
                
                # Extract hand landmarks (21 landmarks per hand, 2 hands max = 42 landmarks)
                frame_landmarks = []
                
                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        hand_coords = []
                        for landmark in hand_landmarks.landmark:
                            # Normalize coordinates to [0,1] relative to frame size
                            hand_coords.extend([landmark.x, landmark.y, landmark.z])
                        frame_landmarks.extend(hand_coords)
                
                # Pad to consistent size (2 hands * 21 landmarks * 3 coords = 126 features)
                # If no hands detected or fewer than 2 hands, pad with zeros
                while len(frame_landmarks) < 126:
                    frame_landmarks.append(0.0)
                
                # Truncate if somehow more than 126 features
                frame_landmarks = frame_landmarks[:126]
                hand_landmarks_seq.append(frame_landmarks)
            else:
                # If MediaPipe not available, create dummy landmarks
                hand_landmarks_seq.append([0.0] * 126)
        
        cap.release()
        
        # Clean up MediaPipe
        if MEDIAPIPE_AVAILABLE:
            hands.close()
        
        # Adjust sequence length for both video and hand landmarks
        if len(frames) == 0:
            return (video_path, None, None, False)
        
        # Pad or trim to target length
        if len(frames) > sequence_length:
            # Take evenly spaced frames
            indices = np.linspace(0, len(frames) - 1, sequence_length).astype(int)
            frames = [frames[i] for i in indices]
            hand_landmarks_seq = [hand_landmarks_seq[i] for i in indices]
        elif len(frames) < sequence_length:
            # Pad with last frame/landmarks
            while len(frames) < sequence_length:
                frames.append(frames[-1])
                hand_landmarks_seq.append(hand_landmarks_seq[-1])
        
        video_seq = np.array(frames) / 255.0
        hand_landmarks_seq = np.array(hand_landmarks_seq)
        
        # Cache the results
        try:
            cached_data = {
                'video_seq': video_seq,
                'hand_landmarks': hand_landmarks_seq,
                'sequence_length': sequence_length,
                'input_size': input_size
            }
            with open(cache_path, 'wb') as f:
                pickle.dump(cached_data, f)
        except:
            pass  # Ignore cache save errors
        
        return (video_path, video_seq, hand_landmarks_seq, False)
        
    except Exception as e:
        print(f"Error processing {video_path}: {e}")
        return (video_path, None, None, False)

class SASLVideoDataset(Dataset):
    """PyTorch Dataset for SASL video sequences with hand landmarks"""
    
    def __init__(self, video_sequences, hand_landmarks, labels, transform=None, augment_factor=0):
        self.video_sequences = video_sequences
        self.hand_landmarks = hand_landmarks
        self.labels = labels
        self.transform = transform
        self.augment_factor = augment_factor
        
        # Apply augmentation if requested
        if augment_factor > 0:
            self._create_augmented_data()
    
    def _create_augmented_data(self):
        """Create augmented versions of the dataset"""
        print(f"Creating augmented data ({self.augment_factor}x multiplier)...")
        
        original_count = len(self.video_sequences)
        augmented_videos = list(self.video_sequences)
        augmented_hands = list(self.hand_landmarks)
        augmented_labels = list(self.labels)
        
        for i in tqdm(range(original_count), desc="Augmenting data"):
            video_seq = self.video_sequences[i]
            hand_seq = self.hand_landmarks[i]
            label = self.labels[i]
            
            for _ in range(self.augment_factor):
                aug_video, aug_hands = self._augment_sequence_pair(video_seq, hand_seq)
                augmented_videos.append(aug_video)
                augmented_hands.append(aug_hands)
                augmented_labels.append(label)
        
        self.video_sequences = augmented_videos
        self.hand_landmarks = augmented_hands
        self.labels = augmented_labels
        
        print(f"Augmentation complete: {original_count} -> {len(self.video_sequences)} videos")
    
    def _augment_sequence_pair(self, video_sequence, hand_sequence):
        """Augment video frames and hand landmarks consistently for one sample."""
        vid = video_sequence.copy()
        hands = hand_sequence.copy()
        seq_len, h, w, c = vid.shape

        aug_types = [
            "brightness_contrast",
            "color_jitter",
            "rotation",
            "gaussian_noise",
            "gaussian_blur",
            "temporal",
            "random_erasing",
            "landmark_affine",
        ]
        augmentation_type = np.random.choice(aug_types)

        if augmentation_type == "brightness_contrast":
            b = np.random.uniform(0.85, 1.15)
            vid = np.clip(vid * b, 0.0, 1.0)
            if np.random.rand() < 0.8:
                cf = np.random.uniform(0.85, 1.15)
                mean = np.mean(vid, axis=(1, 2, 3), keepdims=True)
                vid = np.clip((vid - mean) * cf + mean, 0.0, 1.0)

        elif augmentation_type == "color_jitter":
            sat_factor = np.random.uniform(0.85, 1.15)
            hue_shift = np.random.uniform(-5, 5)
            for i in range(seq_len):
                frame = (vid[i] * 255.0).astype(np.uint8)
                hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
                hsv[:, :, 1] = np.clip(hsv[:, :, 1].astype(np.float32) * sat_factor, 0, 255).astype(np.uint8)
                hsv[:, :, 0] = (hsv[:, :, 0].astype(np.int16) + int(hue_shift)) % 180
                frame_j = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
                vid[i] = frame_j.astype(np.float32) / 255.0

        elif augmentation_type == "rotation":
            angle = float(np.random.uniform(-5.0, 5.0))
            center = (w // 2, h // 2)
            rot = cv2.getRotationMatrix2D(center, angle, 1.0)
            for i in range(seq_len):
                vid[i] = cv2.warpAffine(vid[i], rot, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

        elif augmentation_type == "gaussian_noise":
            std = float(np.random.uniform(0.005, 0.02))
            noise = np.random.normal(0, std, vid.shape).astype(np.float32)
            vid = np.clip(vid + noise, 0.0, 1.0)

        elif augmentation_type == "gaussian_blur":
            k = np.random.choice([3, 5])
            for i in range(seq_len):
                vid[i] = cv2.GaussianBlur(vid[i], (k, k), 0)

        elif augmentation_type == "temporal":
            drop_ratio = np.random.uniform(0.05, 0.15)
            n_drop = max(1, int(seq_len * drop_ratio))
            drop_idx = set(np.random.choice(seq_len, n_drop, replace=False).tolist())
            keep_idx = [i for i in range(seq_len) if i not in drop_idx]
            if not keep_idx:
                keep_idx = list(range(seq_len))
            vid = vid[keep_idx]
            hands = hands[keep_idx]
            while len(vid) < seq_len:
                vid = np.append(vid, [vid[-1]], axis=0)
                hands = np.append(hands, [hands[-1]], axis=0)

        elif augmentation_type == "random_erasing":
            area_ratio = np.random.uniform(0.02, 0.07)
            erase_frames = np.random.choice(seq_len, size=max(1, seq_len // 6), replace=False)
            erase_h = max(1, int(h * np.sqrt(area_ratio) * np.random.uniform(0.8, 1.2)))
            erase_w = max(1, int(w * np.sqrt(area_ratio) * np.random.uniform(0.8, 1.2)))
            for i in erase_frames:
                y = np.random.randint(0, max(1, h - erase_h + 1))
                x = np.random.randint(0, max(1, w - erase_w + 1))
                val = np.random.uniform(0.0, 1.0)
                vid[i, y:y+erase_h, x:x+erase_w, :] = val

        elif augmentation_type == "landmark_affine":
            scale = np.random.uniform(0.97, 1.03)
            rot_deg = np.random.uniform(-3.0, 3.0)
            rot_rad = np.deg2rad(rot_deg)
            cos_r, sin_r = np.cos(rot_rad), np.sin(rot_rad)
            hands_r = hands.reshape(hands.shape[0], 2, 21, 3)
            xy = hands_r[..., :2]
            xy_centered = xy - 0.5
            x_new = (xy_centered[..., 0] * cos_r - xy_centered[..., 1] * sin_r) * scale
            y_new = (xy_centered[..., 0] * sin_r + xy_centered[..., 1] * cos_r) * scale
            xy_j = np.stack([x_new, y_new], axis=-1) + 0.5
            xy_j = np.clip(xy_j, 0.0, 1.0)
            hands_r[..., :2] = xy_j
            if np.random.rand() < 0.7:
                z_noise = np.random.normal(0, 0.005, hands_r[..., 2].shape)
                hands_r[..., 2] = np.clip(hands_r[..., 2] + z_noise, -1.0, 1.0)
            if np.random.rand() < 0.15:
                hand_idx = np.random.choice([0, 1])
                hands_r[:, hand_idx, :, :] = 0.0
            hands = hands_r.reshape(hands.shape[0], -1)

        # Always apply tiny landmark noise sometimes
        if np.random.rand() < 0.6:
            noise_std = np.random.uniform(0.001, 0.006)
            hands = hands + np.random.normal(0, noise_std, hands.shape)
            hands_r2 = hands.reshape(hands.shape[0], 2, 21, 3)
            hands_r2[..., 0:2] = np.clip(hands_r2[..., 0:2], 0.0, 1.0)
            hands_r2[..., 2] = np.clip(hands_r2[..., 2], -1.0, 1.0)
            hands = hands_r2.reshape(hands.shape[0], -1)

        return vid, hands
    
    def __len__(self):
        return len(self.video_sequences)
    
    def __getitem__(self, idx):
        video = torch.FloatTensor(self.video_sequences[idx]).permute(0, 3, 1, 2)  # (seq, C, H, W)
        hands = torch.FloatTensor(self.hand_landmarks[idx])  # (seq, hand_features)
        label = torch.LongTensor([self.labels[idx]])
        
        # Apply ImageNet normalization per frame, per channel
        # video is (seq, C, H, W); broadcast mean/std to (1, C, 1, 1)
        mean = IMAGENET_MEAN.view(1, 3, 1, 1)
        std = IMAGENET_STD.view(1, 3, 1, 1)
        video = (video - mean) / std

        # If additional transforms were provided, apply after normalization per-frame
        if self.transform:
            transformed_frames = []
            for frame in video:
                transformed_frames.append(self.transform(frame))
            video = torch.stack(transformed_frames)
        
        return video, hands, label

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

class HandLandmarkLSTM(nn.Module):
    """LSTM model for hand landmark sequences"""
    
    def __init__(self, num_classes, sequence_length=30, hand_features=126):
        super(HandLandmarkLSTM, self).__init__()
        
        self.sequence_length = sequence_length
        self.hand_features = hand_features  # 2 hands * 21 landmarks * 3 coords
        self.num_classes = num_classes
        
        # Input processing
        self.input_projection = nn.Linear(hand_features, 256)
        self.input_dropout = nn.Dropout(0.2)
        
        # LSTM layers for temporal modeling
        self.lstm1 = nn.LSTM(256, 128, bidirectional=True, batch_first=True, dropout=0.3)
        self.lstm2 = nn.LSTM(256, 64, bidirectional=True, batch_first=True, dropout=0.3)
        
        # Classification layers
        self.classifier = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )
    
    def forward(self, x):
        # x shape: (batch_size, sequence_length, hand_features)
        
        # Project hand features
        x = self.input_projection(x)  # (batch, seq, 256)
        x = torch.relu(x)
        x = self.input_dropout(x)
        
        # LSTM processing
        x, _ = self.lstm1(x)  # (batch, seq, 256)
        x, _ = self.lstm2(x)  # (batch, seq, 128)
        
        # Global average pooling over sequence
        x = torch.mean(x, dim=1)  # (batch, 128)
        
        # Classification
        x = self.classifier(x)
        
        return x

class CombinedCNNHandModel(nn.Module):
    """Combined model that processes both video frames and hand landmarks"""
    
    def __init__(self, num_classes, sequence_length=30, input_size=(224, 224)):
        super(CombinedCNNHandModel, self).__init__()
        
        self.num_classes = num_classes
        
        # CNN branch for video frames
        self.cnn_branch = CNNLSTMModel(num_classes, sequence_length, input_size)
        
        # Hand landmark branch
        self.hand_branch = HandLandmarkLSTM(num_classes, sequence_length)
        
        # Fusion layer to combine predictions
        self.fusion = nn.Sequential(
            nn.Linear(num_classes * 2, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes)
        )
        
        # Learnable weights for combining branches
        self.cnn_weight = nn.Parameter(torch.tensor(0.7))
        self.hand_weight = nn.Parameter(torch.tensor(0.3))
    
    def forward(self, video_frames, hand_landmarks):
        # Get predictions from both branches
        cnn_logits = self.cnn_branch(video_frames)
        hand_logits = self.hand_branch(hand_landmarks)
        
        # Combine logits with learnable weights
        combined_features = torch.cat([cnn_logits, hand_logits], dim=1)
        
        # Final prediction through fusion layer
        final_logits = self.fusion(combined_features)
        
        # Also return individual predictions for analysis
        return final_logits, cnn_logits, hand_logits

class CNNOnlyVideoSASLTrainer:
    """
    CNN-only PyTorch-based video SASL training system
    """
    
    def __init__(self, video_dataset_path="video_dataset", sequence_length=30, input_size=(224, 224), 
                 epochs=100, batch_size=4, augmentation_factor=0, 
                 learning_rate=0.001, num_workers=4):
        """
        Initialize the PyTorch CNN-only video-based SASL trainer
        """
        self.video_dataset_path = Path(video_dataset_path)
        self.sequence_length = sequence_length
        self.input_size = input_size
        self.epochs = epochs
        self.batch_size = batch_size
        self.augmentation_factor = augmentation_factor
        self.learning_rate = learning_rate
        self.num_workers = min(num_workers, mp_cpu.cpu_count())
        self.num_classes = 0
        self.class_names = []
        
        # Create cache directory
        self.cache_dir = Path("outputs/video_cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Validate parameters
        if epochs <= 0:
            raise ValueError("Epochs must be greater than 0")
        if batch_size <= 0:
            raise ValueError("Batch size must be greater than 0")
        if augmentation_factor < 0 or augmentation_factor > 3:
            raise ValueError("Augmentation factor must be between 0 and 3")
        if learning_rate <= 0:
            raise ValueError("Learning rate must be greater than 0")
        
        print(f"CNN-Only VideoSASLTrainer initialized:")
        print(f"  Video dataset: {video_dataset_path}")
        print(f"  Sequence length: {sequence_length} frames")
        print(f"  Input size: {input_size}")
        print(f"  Training epochs: {epochs}")
        print(f"  Batch size: {batch_size}")
        print(f"  Augmentation factor: {augmentation_factor}x")
        print(f"  Learning rate: {learning_rate}")
        print(f"  Device: {device}")
        print(f"  Workers: {self.num_workers}")
        
        # Estimate training time
        estimated_time_mins = epochs * (1 + augmentation_factor) * 0.3
        print(f"  Estimated training time: {estimated_time_mins:.0f}-{estimated_time_mins*2:.0f} minutes")
        
        # Create dataset directory
        self.video_dataset_path.mkdir(exist_ok=True)
    
    def plot_training_history(self, training_history):
        """Generate training and validation accuracy/loss plots"""
        print("\nCreating training history plots...")
        
        # Create figure with subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        epochs = range(1, len(training_history['train_loss']) + 1)
        
        # Plot training and validation loss
        ax1.plot(epochs, training_history['train_loss'], 'bo-', label='Training Loss', linewidth=2, markersize=6)
        ax1.plot(epochs, training_history['val_loss'], 'ro-', label='Validation Loss', linewidth=2, markersize=6)
        ax1.set_title('CNN+LSTM Training and Validation Loss', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Epochs', fontsize=12)
        ax1.set_ylabel('Loss', fontsize=12)
        ax1.legend(fontsize=11)
        ax1.grid(True, alpha=0.3)
        
        # Plot training and validation accuracy
        ax2.plot(epochs, training_history['train_acc'], 'bo-', label='Training Accuracy', linewidth=2, markersize=6)
        ax2.plot(epochs, training_history['val_acc'], 'ro-', label='Validation Accuracy', linewidth=2, markersize=6)
        ax2.set_title('CNN+LSTM Training and Validation Accuracy', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Epochs', fontsize=12)
        ax2.set_ylabel('Accuracy (%)', fontsize=12)
        ax2.legend(fontsize=11)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save the plot
        plot_filename = self.output_dir / "plots" / "cnn_lstm_training_history.png"
        plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  Training history plot saved: {plot_filename}")
        return str(plot_filename)
    
    def generate_confusion_matrix(self, model, data_loader, class_names):
        """Generate confusion matrix for the CNN+LSTM model"""
        print("\nGenerating confusion matrix...")
        
        model.eval()
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for videos, hands, labels_batch in tqdm(data_loader, desc="Evaluating model"):
                videos = videos.to(device)
                hands = hands.to(device)
                # Ensure labels have shape (batch,) instead of (batch,1)
                labels_batch = labels_batch.squeeze().to(device)
                
                final_outputs, _, _ = model(videos, hands)
                _, predicted = torch.max(final_outputs, 1)
                
                # Extend with Python lists of ints to avoid shape ambiguity
                all_predictions.extend(predicted.cpu().tolist())
                all_labels.extend(labels_batch.cpu().tolist())
        
        # Generate confusion matrix
        cm = confusion_matrix(all_labels, all_predictions)
        
        # Create figure
        plt.figure(figsize=(max(10, len(class_names)), max(8, len(class_names))))
        
        # Create heatmap
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=class_names, yticklabels=class_names,
                   square=True, linewidths=0.5, cbar_kws={"shrink": .8})
        
        plt.title('CNN+Hand Landmark Fusion Confusion Matrix', fontsize=16, fontweight='bold', pad=20)
        plt.xlabel('Predicted Label', fontsize=12, fontweight='bold')
        plt.ylabel('True Label', fontsize=12, fontweight='bold')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        
        # Calculate accuracy per class
        accuracy_per_class = cm.diagonal() / cm.sum(axis=1)
        overall_accuracy = np.trace(cm) / np.sum(cm)
        
        info_text = ''
        for i, class_name in enumerate(class_names):
            info_text += f'{class_name}: {accuracy_per_class[i]:.3f}\n'
        
        plt.figtext(0.02, 0.02, info_text, fontsize=10, 
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
        
        plt.tight_layout()
        
        # Save confusion matrix
        cm_filename = self.output_dir / "confusion_matrices" / "cnn_hand_fusion_confusion_matrix.png"
        plt.savefig(cm_filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Generate classification report
        report = classification_report(all_labels, all_predictions, 
                                     target_names=class_names, output_dict=True)
        
        print(f"  Confusion matrix saved: {cm_filename}")
        print(f"  Overall accuracy: {overall_accuracy:.1%}")
        
        return str(cm_filename), cm, report
    
    def save_comprehensive_results(self, training_history, class_names, cm, report, plot_file, cm_file):
        """Save comprehensive training results with all metrics"""
        print("\nSaving comprehensive results...")
        
        # Enhanced results dictionary
        comprehensive_results = {
            # Training history
            'training_history': training_history,
            
            # Model information
            'model_info': {
                'architecture': 'CNN+Hand Landmark Fusion',
                'cnn_backbone': 'EfficientNet-B0',
                'hand_detection': 'MediaPipe Hands',
                'hand_features': '126 (2 hands * 21 landmarks * 3 coords)',
                'fusion_method': 'Learnable weighted combination + MLP',
                'lstm_layers': 2,
                'lstm_hidden_sizes': [256, 128],
                'bidirectional': True,
                'dropout_rates': [0.3, 0.5, 0.3],
                'sequence_length': self.sequence_length,
                'input_size': self.input_size,
                'num_classes': self.num_classes
            },
            
            # Training configuration
            'training_config': {
                'epochs': self.epochs,
                'batch_size': self.batch_size,
                'learning_rate': self.learning_rate,
                'augmentation_factor': self.augmentation_factor,
                'optimizer': 'Adam',
                'scheduler': 'ReduceLROnPlateau',
                'early_stopping_patience': 15,
                'normalization': 'imagenet_mean_std'
            },
            
            # Dataset information
            'dataset_info': {
                'num_classes': self.num_classes,
                'class_names': class_names,
                'total_videos': training_history.get('dataset_size', 0),
                'train_test_split': '80/20',
                'augmentation_applied': self.augmentation_factor > 0
            },
            
            # Final results
            'final_results': {
                'best_accuracy': max(training_history['val_acc']),
                'final_train_loss': training_history['train_loss'][-1],
                'final_val_loss': training_history['val_loss'][-1],
                'final_train_acc': training_history['train_acc'][-1],
                'final_val_acc': training_history['val_acc'][-1],
                'epochs_trained': len(training_history['train_loss'])
            },
            
            # Confusion matrix data
            'confusion_matrix': cm.tolist(),
            
            # Classification report
            'classification_report': report,
            
            # Training environment
            'environment': {
                'device': str(device),
                'torch_version': torch.__version__,
                'cuda_available': torch.cuda.is_available(),
                'gpu_name': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                'training_date': datetime.now().isoformat()
            },
            
            # Output files
            'output_files': {
                'model_file': str(self.output_dir / "models" / "best_sasl_cnn_lstm_model.pth"),
                'plot_file': plot_file,
                'confusion_matrix_file': cm_file,
                'class_names_file': str(self.output_dir / "results" / "class_names.json"),
                'results_file': str(self.output_dir / "results" / "cnn_training_results.json")
            }
        }
        
        # Save comprehensive results
        results_file = self.output_dir / "results" / "cnn_training_results.json"
        with open(results_file, 'w') as f:
            json.dump(comprehensive_results, f, indent=2, default=str)
        
        # Save class names separately (for compatibility)
        classes_file = self.output_dir / "results" / "class_names.json"
        with open(classes_file, 'w') as f:
            json.dump(class_names, f, indent=2)
        
        # Create a detailed summary report
        summary_file = self.output_dir / "results" / "training_summary.txt"
        with open(summary_file, 'w') as f:
            f.write("SASL CNN-Only Training Summary\n")
            f.write("=" * 50 + "\n\n")
            
            f.write(f"Training Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Device Used: {device}\n\n")
            
            f.write("Dataset Information:\n")
            f.write(f"  Classes: {self.num_classes}\n")
            f.write(f"  Class Names: {', '.join(class_names)}\n")
            f.write(f"  Total Videos: {comprehensive_results['dataset_info']['total_videos']}\n")
            f.write(f"  Augmentation Factor: {self.augmentation_factor}x\n\n")
            
            f.write("Training Configuration:\n")
            f.write(f"  Epochs: {self.epochs}\n")
            f.write(f"  Batch Size: {self.batch_size}\n")
            f.write(f"  Learning Rate: {self.learning_rate}\n")
            f.write(f"  Sequence Length: {self.sequence_length}\n")
            f.write(f"  Input Size: {self.input_size}\n\n")
            
            f.write("Final Results:\n")
            f.write(f"  Best Validation Accuracy: {comprehensive_results['final_results']['best_accuracy']:.2f}%\n")
            f.write(f"  Final Training Accuracy: {comprehensive_results['final_results']['final_train_acc']:.2f}%\n")
            f.write(f"  Final Validation Accuracy: {comprehensive_results['final_results']['final_val_acc']:.2f}%\n")
            f.write(f"  Epochs Trained: {comprehensive_results['final_results']['epochs_trained']}\n\n")
            
            f.write("Per-Class Performance:\n")
            for class_name in class_names:
                if class_name in report:
                    precision = report[class_name]['precision']
                    recall = report[class_name]['recall']
                    f1_score = report[class_name]['f1-score']
                    f.write(f"  {class_name}: Precision={precision:.3f}, Recall={recall:.3f}, F1={f1_score:.3f}\n")
            
            f.write(f"\nOutput Files:\n")
            f.write(f"  Model: {comprehensive_results['output_files']['model_file']}\n")
            f.write(f"  Training Plot: {comprehensive_results['output_files']['plot_file']}\n")
            f.write(f"  Confusion Matrix: {comprehensive_results['output_files']['confusion_matrix_file']}\n")
            f.write(f"  Results: {comprehensive_results['output_files']['results_file']}\n")
        
        print(f"  Comprehensive results saved: {results_file}")
        print(f"  Class names saved: {classes_file}")
        print(f"  Training summary saved: {summary_file}")
        
        return str(results_file), str(summary_file)
    
    def load_video_dataset(self):
        """Load all videos with parallel processing"""
        print("\\nLoading video dataset with parallel processing...")
        start_time = time.time()
        
        # Get class directories
        class_dirs = [d for d in self.video_dataset_path.iterdir() if d.is_dir() and d.name != '.gitkeep']
        class_dirs.sort()
        
        if not class_dirs:
            raise ValueError(f"No class directories found in {self.video_dataset_path}")
        
        print(f"Found {len(class_dirs)} SASL classes")
        
        # Collect all video files
        all_video_tasks = []
        class_names = []
        
        for class_idx, class_dir in enumerate(class_dirs):
            class_name = class_dir.name
            class_names.append(class_name)
            
            # Get video files for this class
            video_files = []
            for ext in ['*.mp4', '*.avi', '*.mov', '*.mkv']:
                video_files.extend(class_dir.glob(ext))
            
            print(f"  {class_name}: {len(video_files)} videos")
            
            # Add to processing tasks
            for video_file in video_files:
                task = (video_file, self.sequence_length, self.input_size, self.cache_dir)
                all_video_tasks.append(task)
        
        self.num_classes = len(class_names)
        self.class_names = class_names
        
        print(f"\\nProcessing {len(all_video_tasks)} videos with {self.num_workers} workers...")
        
        video_sequences = []
        hand_landmarks_sequences = []
        labels = []
        cached_count = 0
        processed_count = 0
        
        # Process videos in parallel
        with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
            futures = {executor.submit(process_single_video, task): task for task in all_video_tasks}
            
            with tqdm(total=len(all_video_tasks), desc="Processing videos", 
                     bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]") as pbar:
                
                for future in as_completed(futures):
                    video_path, video_seq, hand_landmarks_seq, was_cached = future.result()
                    
                    if video_seq is not None and hand_landmarks_seq is not None:
                        video_sequences.append(video_seq)
                        hand_landmarks_sequences.append(hand_landmarks_seq)
                        
                        # Determine label from path
                        class_name = Path(video_path).parent.name
                        label = class_names.index(class_name)
                        labels.append(label)
                        
                        if was_cached:
                            cached_count += 1
                        else:
                            processed_count += 1
                    
                    pbar.update(1)
        
        elapsed_time = time.time() - start_time
        
        print(f"\\nDataset loading complete!")
        print(f"  Total time: {elapsed_time:.1f}s")
        print(f"  Cached videos: {cached_count}")
        print(f"  Processed videos: {processed_count}")
        print(f"  Total loaded: {len(video_sequences)} videos")
        print(f"  Hand landmarks extracted: {len(hand_landmarks_sequences)} sequences")
        print(f"  Classes: {len(class_names)}")
        
        return video_sequences, hand_landmarks_sequences, labels, class_names
    
    def create_output_directories(self):
        """Create organized output directory structure"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = Path(f"outputs/training_{timestamp}")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        (self.output_dir / "models").mkdir(exist_ok=True)
        (self.output_dir / "plots").mkdir(exist_ok=True)
        (self.output_dir / "results").mkdir(exist_ok=True)
        (self.output_dir / "confusion_matrices").mkdir(exist_ok=True)
        
        return self.output_dir
    
    def train_model(self):
        """Train CNN+LSTM model"""
        print("\\n" + "="*80)
        print("SASL CNN-ONLY VIDEO TRAINING")
        print("="*80)
        
        # Load dataset
        video_sequences, hand_landmarks_sequences, labels, class_names = self.load_video_dataset()
        
        if len(video_sequences) == 0:
            raise ValueError("No videos loaded. Check your dataset.")
        
        # Create output directories
        output_dir = self.create_output_directories()
        print(f"\\nOutput directory created: {output_dir}")
        
        print(f"\\nDataset Information:")
        print(f"  Video sequences: {len(video_sequences)}")
        print(f"  Classes: {self.num_classes}")
        print(f"  Class names: {', '.join(class_names[:5])}{'...' if len(class_names) > 5 else ''}")
        
        # Split data
        video_train, video_test, hands_train, hands_test, labels_train, labels_test = train_test_split(
            video_sequences, hand_landmarks_sequences, labels, test_size=0.2, random_state=42, stratify=labels
        )
        
        print(f"\\nData split:")
        print(f"  Training: {len(video_train)} videos")
        print(f"  Testing: {len(video_test)} videos")
        
        # Create datasets and data loaders
        print(f"\\nCreating PyTorch datasets with hand landmarks...")
        
        train_dataset = SASLVideoDataset(
            video_train, hands_train, labels_train, augment_factor=self.augmentation_factor
        )
        
        test_dataset = SASLVideoDataset(video_test, hands_test, labels_test)
        
        # Data loaders
        train_loader = DataLoader(
            train_dataset, batch_size=self.batch_size, 
            shuffle=True, num_workers=0, drop_last=True
        )
        
        test_loader = DataLoader(
            test_dataset, batch_size=self.batch_size, 
            shuffle=False, num_workers=0
        )
        
        # Create model
        print(f"\\nCreating Combined CNN+Hand Landmark model...")
        model = CombinedCNNHandModel(self.num_classes, self.sequence_length, self.input_size).to(device)
        
        # Optimizer and loss
        optimizer = optim.Adam(model.parameters(), lr=self.learning_rate)
        criterion = nn.CrossEntropyLoss()
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=8, factor=0.5, min_lr=1e-6)
        
        print(f"Model created and moved to {device}")
        
        # Training results storage
        training_history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
        best_acc = 0.0
        patience_counter = 0
        
    # Training loop (stage 1: head + LSTMs, backbone frozen)
        print(f"\\n" + "="*60)
        print("TRAINING CNN+LSTM MODEL")
        print("="*60)
        
        for epoch in range(self.epochs):
            # Training phase
            model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{self.epochs} [Train]")
            for batch_idx, (videos, hands, labels_batch) in enumerate(train_pbar):
                videos = videos.to(device)
                hands = hands.to(device)
                labels_batch = labels_batch.squeeze().to(device)
                
                optimizer.zero_grad()
                final_outputs, cnn_outputs, hand_outputs = model(videos, hands)
                
                # Calculate loss on final combined output
                loss = criterion(final_outputs, labels_batch)
                
                # Optional: Add auxiliary losses for individual branches
                aux_loss_cnn = criterion(cnn_outputs, labels_batch)
                aux_loss_hand = criterion(hand_outputs, labels_batch)
                total_loss = loss + 0.2 * aux_loss_cnn + 0.1 * aux_loss_hand
                
                total_loss.backward()
                optimizer.step()
                
                train_loss += total_loss.item()
                _, predicted = torch.max(final_outputs.data, 1)
                train_total += labels_batch.size(0)
                train_correct += (predicted == labels_batch).sum().item()
                
                # Update progress bar
                train_acc = 100 * train_correct / train_total
                train_pbar.set_postfix({
                    'Loss': f'{total_loss.item():.4f}',
                    'Acc': f'{train_acc:.1f}%'
                })
            
            # Validation phase
            model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                val_pbar = tqdm(test_loader, desc=f"Epoch {epoch+1}/{self.epochs} [Val]")
                for videos, hands, labels_batch in val_pbar:
                    videos = videos.to(device)
                    hands = hands.to(device)
                    labels_batch = labels_batch.squeeze().to(device)
                    
                    final_outputs, cnn_outputs, hand_outputs = model(videos, hands)
                    
                    # Use combined output for validation
                    loss = criterion(final_outputs, labels_batch)
                    
                    val_loss += loss.item()
                    _, predicted = torch.max(final_outputs.data, 1)
                    val_total += labels_batch.size(0)
                    val_correct += (predicted == labels_batch).sum().item()
                    
                    # Update progress bar
                    val_acc = 100 * val_correct / val_total
                    val_pbar.set_postfix({
                        'Loss': f'{loss.item():.4f}',
                        'Acc': f'{val_acc:.1f}%'
                    })
            
            # Calculate epoch metrics
            epoch_train_loss = train_loss / len(train_loader)
            epoch_train_acc = 100 * train_correct / train_total
            epoch_val_loss = val_loss / len(test_loader)
            epoch_val_acc = 100 * val_correct / val_total
            
            # Store metrics
            training_history['train_loss'].append(epoch_train_loss)
            training_history['train_acc'].append(epoch_train_acc)
            training_history['val_loss'].append(epoch_val_loss)
            training_history['val_acc'].append(epoch_val_acc)
            
            # Learning rate scheduling
            scheduler.step(epoch_val_loss)
            
            # Print epoch results
            print(f"Epoch {epoch+1}/{self.epochs}:")
            print(f"  Train Loss: {epoch_train_loss:.4f}, Train Acc: {epoch_train_acc:.2f}%")
            print(f"  Val Loss: {epoch_val_loss:.4f}, Val Acc: {epoch_val_acc:.2f}%")
            print(f"  LR: {optimizer.param_groups[0]['lr']:.2e}")
            
            # Save best model
            if epoch_val_acc > best_acc:
                best_acc = epoch_val_acc
                torch.save(model.state_dict(), 
                          self.output_dir / "models" / "best_sasl_cnn_lstm_model.pth")
                print(f"  *** New best model saved! Accuracy: {best_acc:.2f}% ***")
                patience_counter = 0
            else:
                patience_counter += 1
            
            # Early stopping
            if patience_counter >= 15:
                print(f"\\nEarly stopping triggered after {patience_counter} epochs without improvement")
                break
        

        # Add dataset size to training history for results
        training_history['dataset_size'] = len(video_sequences)
        
        print(f"\\n" + "="*60)
        print("GENERATING COMPREHENSIVE OUTPUTS")
        print("="*60)
        
        # Generate training history plots
        plot_file = self.plot_training_history(training_history)
        
        # Load best model for confusion matrix generation
        best_model = CombinedCNNHandModel(self.num_classes, self.sequence_length, self.input_size).to(device)
        best_model.load_state_dict(torch.load(self.output_dir / "models" / "best_sasl_cnn_lstm_model.pth"))
        
        # Generate confusion matrix
        cm_file, cm, report = self.generate_confusion_matrix(best_model, test_loader, class_names)
        
        # Save comprehensive results
        results_file, summary_file = self.save_comprehensive_results(
            training_history, class_names, cm, report, plot_file, cm_file
        )
        
        print(f"\\n" + "="*80)
        print("CNN-ONLY TRAINING COMPLETE!")
        print("="*80)
        print(f"Final Results:")
        print(f"  CNN+LSTM Accuracy: {best_acc:.1f}%")
        print(f"  Epochs Trained: {epoch + 1}/{self.epochs}")
        print(f"  Dataset Size: {len(video_sequences)} videos")
        print(f"  Classes: {len(class_names)}")
        print(f"\\nOutput Directory: {self.output_dir}")
        print(f"  Model: best_sasl_cnn_lstm_model.pth")
        print(f"  Training Plot: cnn_lstm_training_history.png")
        print(f"  Confusion Matrix: cnn_hand_fusion_confusion_matrix.png")
        print(f"  Results: cnn_training_results.json")
        print(f"  Summary: training_summary.txt")
        
        return model

if __name__ == "__main__":
    print("SASL CNN-Only Video Training System")
    print("=" * 60)
    
    # Example usage
    trainer = CNNOnlyVideoSASLTrainer(
        video_dataset_path="video_dataset",
        sequence_length=30,
        input_size=(224, 224),
        epochs=50,
        batch_size=4,
        augmentation_factor=1
    )
    
    model = trainer.train_model()
    
    if model is not None:
        print("SUCCESS: CNN-only model trained successfully!")
    else:
        print("Training failed. Check your dataset.")