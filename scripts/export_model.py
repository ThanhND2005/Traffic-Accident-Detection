"""
Model Export Script.
Exports YOLO11, LSTM Trajectory Classifier, and Fusion modules to ONNX / TorchScript for optimized inference.
"""

import os
import argparse
import torch

from src.classifiers.lstm_classifier import TrajectoryAccidentClassifier
from src.fusion.fusion import AccidentFusion


def export_yolo(weights_path: str, format: str = "onnx"):
    """Export YOLO11 model using Ultralytics export."""
    try:
        from ultralytics import YOLO
        print(f"Loading YOLO model: {weights_path}")
        model = YOLO(weights_path)
        output_path = model.export(format=format)
        print(f"✅ YOLO exported successfully to: {output_path}")
    except Exception as e:
        print(f"❌ Failed to export YOLO: {e}")


def export_lstm(weights_path: str, output_path: str = "checkpoints/lstm_classifier.onnx"):
    """Export LSTM Classifier to ONNX format."""
    print(f"Loading LSTM model from: {weights_path}")
    model = TrajectoryAccidentClassifier.load_from_checkpoint(weights_path, device="cpu")
    model.eval()

    dummy_input = torch.randn(1, 30, 15, dtype=torch.float32)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=["trajectory_seq"],
        output_names=["accident_prob"],
        dynamic_axes={"trajectory_seq": {0: "batch_size"}, "accident_prob": {0: "batch_size"}},
    )
    print(f"✅ LSTM Classifier exported to ONNX: {output_path}")


def export_fusion(weights_path: str, output_path: str = "checkpoints/fusion.onnx"):
    """Export Fusion MLP to ONNX format."""
    model = AccidentFusion(method="mlp", weights_path=weights_path)
    model.eval()

    dummy_motion = torch.randn(1, 1, dtype=torch.float32)
    dummy_visual = torch.randn(1, 1, dtype=torch.float32)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    torch.onnx.export(
        model,
        (dummy_motion, dummy_visual),
        output_path,
        export_params=True,
        opset_version=14,
        input_names=["motion_score", "visual_score"],
        output_names=["fused_score"],
    )
    print(f"✅ Fusion module exported to ONNX: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export models to ONNX/TorchScript")
    parser.add_argument("--model", type=str, choices=["yolo", "lstm", "fusion", "all"], default="yolo")
    parser.add_argument("--weights", type=str, default="yolo11s.pt", help="Path to input weights")
    parser.add_argument("--out", type=str, default=None, help="Output file path")
    args = parser.parse_args()

    if args.model == "yolo" or args.model == "all":
        export_yolo(args.weights)
    if args.model == "lstm" or args.model == "all":
        out = args.out or "checkpoints/lstm_classifier.onnx"
        if os.path.exists(args.weights):
            export_lstm(args.weights, out)
    if args.model == "fusion" or args.model == "all":
        out = args.out or "checkpoints/fusion.onnx"
        export_fusion(args.weights if os.path.exists(args.weights) else None, out)
