"""
Video frame loading for Xperience-10M.
"""

import os
from dataclasses import dataclass
from pathlib import Path

import cv2


VIDEO_EXTENSIONS = (".mp4", ".mov", ".avi", ".mkv", ".webm")


@dataclass(frozen=True)
class VideoMetadata:
    path: Path
    width: int
    height: int
    fps: float
    num_frames: int

    @property
    def duration_seconds(self):
        return self.num_frames / self.fps if self.fps > 0 else 0.0


def discover_video_files(root, extensions=VIDEO_EXTENSIONS):
    """Return sorted video files under root."""
    root_path = Path(root)
    if root_path.is_file():
        return [root_path]

    normalized_exts = {ext.lower() for ext in extensions}
    return sorted(
        path
        for path in root_path.rglob("*")
        if path.is_file() and path.suffix.lower() in normalized_exts
    )


def get_video_metadata(video_path):
    """Return basic video metadata."""
    path = Path(video_path)
    if not path.exists():
        raise FileNotFoundError(f"Video not found: {path}")

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        cap.release()
        raise RuntimeError(f"Unable to open video: {path}")

    metadata = VideoMetadata(
        path=path,
        width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        fps=float(cap.get(cv2.CAP_PROP_FPS) or 0.0),
        num_frames=int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0),
    )
    cap.release()
    return metadata


def iter_video_frames(video_path, start_frame=0, end_frame=None, frame_stride=1):
    """Yield selected RGB frames as (frame_idx, frame_rgb)."""
    if frame_stride <= 0:
        raise ValueError("frame_stride must be >= 1")

    path = Path(video_path)
    if not path.exists():
        raise FileNotFoundError(f"Video not found: {path}")

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        cap.release()
        raise RuntimeError(f"Unable to open video: {path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if start_frame < 0:
        start_frame = 0
    if end_frame is None or end_frame < 0 or end_frame > total_frames:
        end_frame = total_frames
    if start_frame >= end_frame:
        cap.release()
        return

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    frame_idx = start_frame

    try:
        while frame_idx < end_frame:
            ret, frame = cap.read()
            if not ret:
                break
            if (frame_idx - start_frame) % frame_stride == 0:
                yield frame_idx, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_idx += 1
    finally:
        cap.release()


def load_video_frame(video_path, frame_idx, log_image_scale=1.0):
    """Load a single frame from a video. Returns (H, W, 3) RGB or None."""
    if not os.path.exists(video_path):
        return None
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        return None
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    if log_image_scale != 1.0:
        frame_rgb = cv2.resize(frame_rgb, (int(frame_rgb.shape[1] * log_image_scale), int(frame_rgb.shape[0] * log_image_scale)))
    return frame_rgb


__all__ = [
    "VIDEO_EXTENSIONS",
    "VideoMetadata",
    "discover_video_files",
    "get_video_metadata",
    "iter_video_frames",
    "load_video_frame",
]
