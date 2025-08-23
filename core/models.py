from dataclasses import dataclass
from typing import List, Dict, Optional
import numpy as np


@dataclass
class VideoFrame:
    index: int
    timestamp: float
    image: np.ndarray
    features: np.ndarray

    @property
    def width(self) -> int:
        return self.image.shape[1]

    @property
    def height(self) -> int:
        return self.image.shape[0]


@dataclass
class VideoNode:
    id: int
    frame_indices: List[int]
    children: List[int]
    parent: Optional[int] = None
    level: int = 0
    representative_frame_idx: Optional[int] = None


@dataclass
class VideoTree:
    nodes: Dict[int, VideoNode]
    frames: List[VideoFrame]

    def get_node_frames(self, node_id: int) -> List[VideoFrame]:
        return [self.frames[i] for i in self.nodes[node_id].frame_indices]

    def get_representative_frame(self, node_id: int) -> VideoFrame:
        node = self.nodes[node_id]
        if node.representative_frame_idx is not None:
            return self.frames[node.representative_frame_idx]
        return self.frames[node.frame_indices[0]]