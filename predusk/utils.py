import json
import time
from pathlib import Path
from contextlib import contextmanager
import cv2
import numpy as np


def get_video_info(video_path: Path) -> dict:
    cap = cv2.VideoCapture(str(video_path))
    info = {
        "fps": cap.get(cv2.CAP_PROP_FPS),
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        "duration_seconds": 0,
    }
    if info["fps"] > 0:
        info["duration_seconds"] = round(info["frame_count"] / info["fps"], 2)
    cap.release()
    return info


def format_stats(tracking_summary: dict) -> str:
    lines = [
        "### Tracking Summary",
        f"- **Total Frames Processed:** {tracking_summary.get('total_frames', 0)}",
        f"- **Unique Player IDs Assigned:** {tracking_summary.get('unique_ids', 0)}",
    ]

    frame_stats = tracking_summary.get("frame_stats", [])
    if frame_stats:
        counts = [f["count"] for f in frame_stats]
        avg = round(sum(counts) / len(counts), 1)
        peak = max(counts)
        lines.append(f"- **Average Active Players/Frame:** {avg}")
        lines.append(f"- **Peak Players in Single Frame:** {peak}")

    return "\n".join(lines)


def ensure_h264_compatible(input_path: Path, output_path: Path) -> Path:
    try:
        cap = cv2.VideoCapture(str(input_path))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            writer.write(frame)
        cap.release()
        writer.release()
        return output_path
    except Exception:
        return input_path


def save_tracking_report(summary: dict, report_path: Path):
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w") as f:
        json.dump(summary, f, indent=2)


@contextmanager
def timer(label: str = ""):
    start = time.time()
    yield
    elapsed = time.time() - start
    print(f"{label} completed in {elapsed:.2f}s")


def extract_thumbnail(video_path: Path, output_path: Path, frame_index: int = 0) -> Path:
    cap = cv2.VideoCapture(str(video_path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ret, frame = cap.read()
    cap.release()
    if ret:
        cv2.imwrite(str(output_path), frame)
    return output_path
