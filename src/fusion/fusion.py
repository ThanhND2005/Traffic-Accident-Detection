"""
Multi-Modal Accident Fusion Module.
Combines scores from the kinematic/motion branch (Rule-based or LSTM) and visual branch (X3D-S).
"""

import os
from typing import Optional
import torch
import torch.nn as nn


class AccidentFusion(nn.Module):
    """
    Late-fusion module for fusing kinematic motion anomaly scores with visual video classifier scores.
    Supports weighted score fusion and learnable 2-layer MLP fusion.
    """

    def __init__(
        self,
        method: str = "cascade",
        motion_weight: float = 0.6,
        visual_weight: float = 0.4,
        weights_path: Optional[str] = None,
    ):
        super().__init__()
        self.method = method  # "cascade", "weighted", or "mlp"
        self.motion_weight = nn.Parameter(torch.tensor(motion_weight), requires_grad=(method == "weighted"))
        self.visual_weight = nn.Parameter(torch.tensor(visual_weight), requires_grad=(method == "weighted"))

        self.fusion_mlp = nn.Sequential(
            nn.Linear(2, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid(),
        )

        if weights_path and os.path.exists(weights_path):
            try:
                state_dict = torch.load(weights_path, map_location="cpu")
                self.load_state_dict(state_dict)
            except Exception as e:
                print(f"[Warning] Failed to load Fusion weights from {weights_path}: {e}")

    def forward(self, motion_score: torch.Tensor, visual_score: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        Args:
            motion_score: Tensor of shape (batch, 1)
            visual_score: Tensor of shape (batch, 1)
        """
        if self.method == "weighted":
            total_w = torch.abs(self.motion_weight) + torch.abs(self.visual_weight) + 1e-6
            w_m = torch.abs(self.motion_weight) / total_w
            w_v = torch.abs(self.visual_weight) / total_w
            return w_m * motion_score + w_v * visual_score
        else:
            combined = torch.cat([motion_score, visual_score], dim=-1)
            return self.fusion_mlp(combined)

    def fuse(self, motion_score: float, visual_score: Optional[float] = None) -> float:
        """
        Inference fusion function for a single event.

        Args:
            motion_score: Kinematic confidence score [0.0, 1.0].
            visual_score: Optional visual confidence score [0.0, 1.0]. If None, returns motion score.

        Returns:
            Fused confidence score [0.0, 1.0].
        """
        if visual_score is None:
            # Cascade: Visual branch was not triggered
            return float(motion_score)

        if self.method == "cascade" or self.method == "weighted":
            mw = float(self.motion_weight.data)
            vw = float(self.visual_weight.data)
            norm = mw + vw + 1e-6
            fused = (mw * motion_score + vw * visual_score) / norm
            return float(fused)

        # MLP mode
        self.eval()
        with torch.no_grad():
            m_tensor = torch.tensor([[motion_score]], dtype=torch.float32)
            v_tensor = torch.tensor([[visual_score]], dtype=torch.float32)
            fused = self.forward(m_tensor, v_tensor).item()
        return float(fused)

    def save_checkpoint(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.state_dict(), path)

    @classmethod
    def load_from_checkpoint(cls, path: str) -> "AccidentFusion":
        model = cls(weights_path=path)
        return model
