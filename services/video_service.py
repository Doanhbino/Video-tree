from typing import Optional
from core.video_processor import VideoProcessor
from core.tree_builder import VideoTreeBuilder
from core.models import VideoTree
from config import settings
import os
import tempfile
import logging

logger = logging.getLogger(__name__)


class VideoService:
    def __init__(self):
        self.processor = VideoProcessor()
        self.tree_builder = VideoTreeBuilder()

    def process_video_file(self, file_bytes: bytes, file_ext: str = "mp4") -> VideoTree:
        """Process uploaded video file"""
        try:
            with tempfile.NamedTemporaryFile(suffix=f".{file_ext}", delete=False) as tmp_file:
                tmp_file.write(file_bytes)
                tmp_path = tmp_file.name

            try:
                frames = self.processor.process_video(tmp_path)
                video_tree = self.tree_builder.build_tree(frames)
                return video_tree
            finally:
                os.unlink(tmp_path)
        except Exception as e:
            logger.error(f"Video processing failed: {str(e)}")
            raise

    def get_representative_frames(self, video_tree: VideoTree, max_frames: int = 10) -> List[VideoFrame]:
        """Get key representative frames from the tree"""
        nodes = sorted(
            video_tree.nodes.values(),
            key=lambda n: len(n.frame_indices),
            reverse=True
        )

        frames = []
        for node in nodes[:max_frames]:
            if node.representative_frame_idx is not None:
                frames.append(video_tree.frames[node.representative_frame_idx])

        return frames