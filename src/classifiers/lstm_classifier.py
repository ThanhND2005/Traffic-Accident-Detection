"""
Trajectory Accident Classifier (Bidirectional LSTM with Temporal Soft Attention).
Classifies 15-dimensional kinematic feature sequences into accident probabilities.
"""

import os
from typing import Optional
import torch
import torch.nn as nn
import numpy as np


class TrajectoryAccidentClassifier(nn.Module):
    """
    Bidirectional LSTM with Attention Mechanism for Accident Prediction from Trajectory Sequences.
    """

    def __init__(
        self,
        input_dim: int = 15,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=True,
        )

        self.attention = nn.Sequential(
            nn.Linear(hidden_dim * 2, 1),
            nn.Softmax(dim=1),
        )

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (batch_size, seq_len, input_dim)

        Returns:
            Tensor of shape (batch_size, 1) with values in [0, 1] representing accident probability.
        """
        # lstm_out: (batch, seq_len, hidden_dim * 2)
        lstm_out, _ = self.lstm(x)

        # attn_weights: (batch, seq_len, 1)
        attn_weights = self.attention(lstm_out)

        # context: (batch, hidden_dim * 2)
        context = torch.sum(lstm_out * attn_weights, dim=1)

        # output: (batch, 1)
        return self.classifier(context)

    def predict_proba(self, sequence: np.ndarray, device: str = "cpu") -> float:
        """
        Inference on a single trajectory sequence.

        Args:
            sequence: (seq_len, input_dim) numpy array.
            device: 'cpu' or 'cuda'.

        Returns:
            float probability between 0.0 and 1.0.
        """
        self.eval()
        with torch.no_grad():
            tensor_seq = torch.from_numpy(sequence).float().unsqueeze(0).to(device)
            prob = self.forward(tensor_seq).item()
        return float(prob)

    def save_checkpoint(self, path: str):
        """Save model weights and architecture configuration."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(
            {
                "state_dict": self.state_dict(),
                "config": {
                    "input_dim": self.input_dim,
                    "hidden_dim": self.hidden_dim,
                    "num_layers": self.num_layers,
                },
            },
            path,
        )

    @classmethod
    def load_from_checkpoint(cls, path: str, device: str = "cpu") -> "TrajectoryAccidentClassifier":
        """Load model from saved checkpoint."""
        checkpoint = torch.load(path, map_location=device)
        config = checkpoint.get("config", {"input_dim": 15, "hidden_dim": 64, "num_layers": 2})
        model = cls(**config)
        model.load_state_dict(checkpoint["state_dict"])
        model.to(device)
        model.eval()
        return model
