"""
Visual Video Classifier (X3D-S Wrapper).
Operates as the visual branch in the cascade architecture to analyze spatio-temporal video crops (ROI).
"""

import os
from typing import List, Optional, Tuple
import numpy as np
import cv2
import torch
import torch.nn as nn


class Lightweight3DCNN(nn.Module):
    """
    Lightweight fallback 3D CNN architecture for spatio-temporal video classification
    when PyTorchVideo or internet access for torch.hub is unavailable.
    """

    def __init__(self, in_channels: int = 3, num_classes: int = 1):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv3d(in_channels, 32, kernel_size=(3, 3, 3), stride=(1, 2, 2), padding=(1, 1, 1)),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2)),
            nn.Conv3d(32, 64, kernel_size=(3, 3, 3), stride=(1, 2, 2), padding=(1, 1, 1)),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool3d((1, 1, 1)),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(64, num_classes),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (B, C, T, H, W)
        feat = self.features(x)
        feat = feat.view(feat.size(0), -1)
        return self.classifier(feat)


class VideoAccidentClassifier:
    """
    X3D-S / 3D CNN visual classifier for detecting accident presence within video clips.
    """

    def __init__(
        self,
        weights_path: Optional[str] = None,
        device: Optional[str] = None,
        num_frames: int = 16,
        crop_size: int = 160,
    ):
        self.num_frames = num_frames
        self.crop_size = crop_size
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.weights_path = weights_path
        self.model = self._build_model()

    def _build_model(self) -> nn.Module:
        """Initializes X3D-S with fine-tuning head, or fallback architecture."""
        try:
            # Attempt to load X3D from torch.hub / pytorchvideo
            model = torch.hub.load("facebookresearch/pytorchvideo", "x3d_s", pretrained=False)
            in_features = model.blocks[5].proj.in_features if hasattr(model.blocks[5].proj, 'in_features') else 2048
            model.blocks[5].proj = nn.Sequential(
                nn.Linear(in_features, 1),
                nn.Sigmoid(),
            )
        except Exception:
            # Fallback lightweight architecture
            model = Lightweight3DCNN(in_channels=3, num_classes=1)

        if self.weights_path and os.path.exists(self.weights_path):
            try:
                state_dict = torch.load(self.weights_path, map_location=self.device)
                model.load_state_dict(state_dict)
            except Exception as e:
                print(f"[Warning] Failed to load X3D weights from {self.weights_path}: {e}")

        model.to(self.device)
        model.eval()
        return model

    def preprocess_clip(self, frames: List[np.ndarray]) -> torch.Tensor:
        """
        Preprocess a list of BGR frame images into normalized tensor (1, C, T, H, W).
        """
        if len(frames) == 0:
            return torch.zeros((1, 3, self.num_frames, self.crop_size, self.crop_size), device=self.device)

        # Uniform temporal sampling to get exactly `num_frames`
        indices = np.linspace(0, len(frames) - 1, self.num_frames).astype(int)
        processed_frames = []
        for idx in indices:
            frame = frames[idx]
            resized = cv2.resize(frame, (self.crop_size, self.crop_size))
            # Convert BGR to RGB, normalize to [0, 1]
            rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            # Normalize with ImageNet mean and std
            mean = np.array([0.45, 0.45, 0.45], dtype=np.float32)
            std = np.array([0.225, 0.225, 0.225], dtype=np.float32)
            rgb = (rgb - mean) / std
            processed_frames.append(rgb)

        # (T, H, W, C) -> (C, T, H, W)
        tensor = np.stack(processed_frames, axis=0)  # (T, H, W, C)
        tensor = np.transpose(tensor, (3, 0, 1, 2))  # (C, T, H, W)
        tensor = torch.from_numpy(tensor).float().unsqueeze(0).to(self.device)
        return tensor

    def predict(self, frames: List[np.ndarray]) -> float:
        """
        Inference on video clip.

        Returns:
            Probability of accident in [0, 1].
        """
        self.model.eval()
        with torch.no_grad():
            tensor = self.preprocess_clip(frames)
            out = self.model(tensor)
            prob = float(out.squeeze().cpu().item())
        return prob
