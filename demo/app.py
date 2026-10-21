"""
Interactive Gradio Web Application for Traffic Accident Detection.
Allows users to upload traffic videos, adjust detection parameters in real-time,
and inspect flagged accident events with timestamps, severity levels, and bounding box highlights.
"""

import os
import sys
import asyncio
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Set a dedicated local temp folder to avoid Windows AppData locking issues
temp_cache = PROJECT_ROOT / "temp" / "gradio_cache"
temp_cache.mkdir(parents=True, exist_ok=True)
os.environ["GRADIO_TEMP_DIR"] = str(temp_cache)

# Fix Windows file-locking race condition on video upload & Starlette/AnyIO responses
if sys.platform == "win32":
    try:
        from asyncio.proactor_events import _ProactorBasePipeTransport
        _orig_call_connection_lost = _ProactorBasePipeTransport._call_connection_lost

        def _call_connection_lost_patched(self, exc=None):
            try:
                _orig_call_connection_lost(self, exc)
            except (ConnectionResetError, OSError):
                pass

        _ProactorBasePipeTransport._call_connection_lost = _call_connection_lost_patched
    except Exception:
        pass

    try:
        import anyio._core._fileio
        import starlette.responses

        _orig_anyio_open = anyio._core._fileio.open_file

        async def _retry_anyio_open(file, *args, **kwargs):
            max_retries = 20
            delay = 0.05
            for attempt in range(max_retries):
                try:
                    return await _orig_anyio_open(file, *args, **kwargs)
                except PermissionError:
                    if attempt < max_retries - 1:
                        await asyncio.sleep(delay)
                        delay *= 1.3
                    else:
                        raise

        anyio._core._fileio.open_file = _retry_anyio_open
        anyio.open_file = _retry_anyio_open
        starlette.responses.anyio.open_file = _retry_anyio_open
    except Exception:
        pass

import subprocess
import tempfile
import cv2
import gradio as gr
import pandas as pd
import yaml

from src.pipeline import AccidentDetectionPipeline


def convert_to_browser_h264(raw_mp4_path: str) -> str:
    """
    Convert raw OpenCV mp4v video into browser-compatible H.264 video with yuv420p & faststart.
    """
    if not os.path.exists(raw_mp4_path) or os.path.getsize(raw_mp4_path) == 0:
        return raw_mp4_path

    h264_out = tempfile.NamedTemporaryFile(suffix="_h264.mp4", delete=False)
    h264_path = h264_out.name
    h264_out.close()

    try:
        import imageio_ffmpeg
        ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
        cmd = [
            ffmpeg_exe, "-y",
            "-i", raw_mp4_path,
            "-vcodec", "libx264",
            "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",
            h264_path
        ]
        res = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if res.returncode == 0 and os.path.exists(h264_path) and os.path.getsize(h264_path) > 0:
            try:
                os.remove(raw_mp4_path)
            except OSError:
                pass
            return h264_path
    except Exception as e:
        print(f"[Warning] Video conversion to H.264 failed ({e}). Returning raw video.")

    return raw_mp4_path


def process_uploaded_video(
    video_path: str,
    conf_threshold: float,
    rule_threshold: float,
    cooldown_sec: float,
    progress=gr.Progress(),
):
    """
    Process video with user-selected parameters.
    """
    if not video_path or not os.path.exists(video_path):
        return None, pd.DataFrame(columns=["Event ID", "Time", "Level", "Score", "Reasons"]), "⚠️ Please upload a valid video."

    # Load and adjust base config
    config_file = PROJECT_ROOT / "configs" / "default.yaml"
    with open(config_file, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    config["detector"]["conf_threshold"] = conf_threshold
    config["classifiers"]["rule_based"]["alert_threshold"] = rule_threshold
    config["postprocessing"]["cooldown_seconds"] = cooldown_sec

    pipeline = AccidentDetectionPipeline(config)

    temp_out = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    temp_out_path = temp_out.name
    temp_out.close()

    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    pipeline.fps = fps

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(temp_out_path, fourcc, fps, (width, height))

    frame_idx = 0
    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            annotated, alert = pipeline.process_frame(frame, frame_idx)
            writer.write(annotated)

            if frame_idx % 10 == 0:
                progress(frame_idx / total_frames, desc=f"Processing frame {frame_idx}/{total_frames}...")

            frame_idx += 1
    finally:
        cap.release()
        writer.release()

    events = pipeline.post_processor.get_summary()

    # Convert to browser-compatible H.264 format
    try:
        progress(1.0, desc="Encoding video for web playback...")
    except Exception:
        pass
    final_video_path = convert_to_browser_h264(temp_out_path)

    # Build report dataframe
    rows = []
    for ev in events:
        rows.append({
            "Event ID": ev["event_id"],
            "Time": ev["timestamp_str"],
            "Level": ev["level"],
            "Score": f"{ev['score']:.2f}",
            "Reasons": ev["reasons"],
        })

    df = pd.DataFrame(rows)
    summary_msg = f"✅ Processing completed: {frame_idx} frames analyzed. Found {len(events)} confirmed incidents."

    return final_video_path, df, summary_msg


def build_app():
    custom_css = """
    .alert-header { text-align: center; color: #1e3a8a; }
    """

    with gr.Blocks(title="Traffic Accident Detection System") as demo:
        gr.Markdown(
            """
            # 🚦 Automated Traffic Accident Detection & Alert System
            ### Multi-Modal AI Pipeline: YOLO11 + ByteTrack + Kinematic Analysis + Cascade Visual Verification
            """
        )

        with gr.Row():
            with gr.Column(scale=4):
                input_video = gr.Video(label="Input Traffic Video (MP4 / AVI)")
                with gr.Accordion("⚙️ Pipeline Sensitivity & Threshold Settings", open=False):
                    conf_slider = gr.Slider(
                        minimum=0.1, maximum=0.9, value=0.4, step=0.05,
                        label="YOLO Vehicle Detection Confidence"
                    )
                    rule_slider = gr.Slider(
                        minimum=0.2, maximum=0.9, value=0.5, step=0.05,
                        label="Collision Anomaly Alert Threshold"
                    )
                    cooldown_slider = gr.Slider(
                        minimum=1.0, maximum=15.0, value=5.0, step=1.0,
                        label="Alert Cooldown Interval (seconds)"
                    )
                btn_run = gr.Button("🚀 Run Accident Analysis", variant="primary")

            with gr.Column(scale=5):
                output_video = gr.Video(label="Annotated Video with Incident Tracking")
                status_box = gr.Textbox(label="Status Summary", interactive=False)
                events_table = gr.DataFrame(
                    headers=["Event ID", "Time", "Level", "Score", "Reasons"],
                    label="🚨 Incident Log",
                    interactive=False,
                )

        btn_run.click(
            fn=process_uploaded_video,
            inputs=[input_video, conf_slider, rule_slider, cooldown_slider],
            outputs=[output_video, events_table, status_box],
        )

    return demo


if __name__ == "__main__":
    app = build_app()
    app.queue(default_concurrency_limit=2)
    app.launch(server_name="0.0.0.0", server_port=7860, share=False)
