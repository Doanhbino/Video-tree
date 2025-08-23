import decord
import numpy as np
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from typing import List, Tuple
from config import settings
from core.models import VideoFrame
import logging

logger = logging.getLogger(__name__)


class VideoProcessor:
    def __init__(self, model_name: str = "openai/clip-vit-large-patch14"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Initializing CLIP model on {self.device}")
        self.model = CLIPModel.from_pretrained(model_name).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(model_name)

    def _resize_frame(self, frame: np.ndarray) -> np.ndarray:
        """Resize frame to maintain aspect ratio with max width"""
        if frame.shape[1] > settings.MAX_FRAME_WIDTH:
            height = int((settings.MAX_FRAME_WIDTH / frame.shape[1]) * frame.shape[0])
            return cv2.resize(frame, (settings.MAX_FRAME_WIDTH, height))
        return frame

    def extract_frames(self, video_path: str, frame_rate: float = settings.DEFAULT_FRAME_RATE) -> List[VideoFrame]:
        """Extract frames from video with timestamps"""
        try:
            vr = decord.VideoReader(video_path)
            fps = vr.get_avg_fps()
            frame_step = int(fps / frame_rate)

            frames = []
            for i in range(0, len(vr), frame_step):
                frame = vr[i].asnumpy()
                frame = self._resize_frame(frame)
                timestamp = i / fps
                frames.append((frame, timestamp))

            return frames
        except Exception as e:
            logger.error(f"Error reading video: {str(e)}")
            raise

    def extract_features(self, frames: List[np.ndarray]) -> np.ndarray:
        """Extract CLIP features from frames"""
        try:
            images = [Image.fromarray(frame) for frame in frames]
            inputs = self.processor(images=images, return_tensors="pt", padding=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                features = self.model.get_image_features(**inputs)

            return features.cpu().numpy()
        except Exception as e:
            logger.error(f"Feature extraction failed: {str(e)}")
            raise

    def process_video(self, video_path: str, frame_rate: float = settings.DEFAULT_FRAME_RATE) -> List[VideoFrame]:
        """Full video processing pipeline"""
        frame_data = self.extract_frames(video_path, frame_rate)
        frames = [f[0] for f in frame_data]
        timestamps = [f[1] for f in frame_data]

        features = self.extract_features(frames)

        video_frames = []
        for idx, (frame, timestamp) in enumerate(frame_data):
            video_frames.append(
                VideoFrame(
                    index=idx,
                    timestamp=timestamp,
                    image=frame,
                    features=features[idx]
                )
            )

        return video_frames