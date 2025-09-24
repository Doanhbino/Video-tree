from __future__ import annotations
import cv2
import math
import os
from typing import List, Tuple
from dataclasses import dataclass
from ..config import settings

@dataclass
class FrameInfo:
    idx: int
    time_s: float
    path: str

def extract_frames(video_path: str, out_dir: str | None = None, target_fps: float | None = None, 
                   max_frames: int | None = None, frame_size: int | None = None) -> list[FrameInfo]:
    """Extract frames at a target FPS and save to disk; returns list of FrameInfo."""
    if out_dir is None:
        out_dir = os.path.join(settings.DATA_DIR, settings.FRAMES_DIRNAME)
    os.makedirs(out_dir, exist_ok=True)

    if target_fps is None:
        target_fps = settings.TARGET_FPS
    if max_frames is None:
        max_frames = settings.MAX_FRAMES
    if frame_size is None:
        frame_size = settings.FRAME_SIZE

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps if fps > 0 else 0

    sample_every = max(int(round(fps / target_fps)) if target_fps > 0 else 1, 1)

    extracted = []
    i = 0
    saved = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if i % sample_every == 0:
            t = (i / fps) if fps > 0 else 0
            h, w = frame.shape[:2]
            side = min(h, w)
            y0 = (h - side) // 2
            x0 = (w - side) // 2
            crop = frame[y0:y0+side, x0:x0+side]
            resized = cv2.resize(crop, (frame_size, frame_size), interpolation=cv2.INTER_AREA)
            fname = f"frame_{len(extracted):06d}.jpg"
            fpath = os.path.join(out_dir, fname)
            cv2.imwrite(fpath, resized, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
            extracted.append(FrameInfo(idx=i, time_s=t, path=fpath))
            saved += 1
            if max_frames is not None and saved >= max_frames:
                break
        i += 1

    cap.release()
    return extracted
