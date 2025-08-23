import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import AgglomerativeClustering
from typing import List, Dict, Tuple
from collections import defaultdict
from core.models import VideoFrame, VideoNode, VideoTree
import logging

logger = logging.getLogger(__name__)


class VideoTreeBuilder:
    def __init__(
            self,
            similarity_threshold: float = settings.DEFAULT_SIMILARITY_THRESHOLD,
            max_children: int = settings.DEFAULT_MAX_CHILDREN,
            temporal_weight: float = 0.3
    ):
        self.similarity_threshold = similarity_threshold
        self.max_children = max_children
        self.temporal_weight = temporal_weight

    def _compute_combined_similarity(self, frames: List[VideoFrame]) -> np.ndarray:
        """Compute similarity matrix combining visual and temporal features"""
        # Visual similarity
        features = np.array([frame.features for frame in frames])
        visual_sim = cosine_similarity(features)

        # Temporal similarity (closer in time = more similar)
        timestamps = np.array([frame.timestamp for frame in frames])
        time_diff = np.abs(timestamps[:, None] - timestamps[None, :])
        max_time_diff = time_diff.max()
        temporal_sim = 1 - (time_diff / max_time_diff)

        # Combined similarity
        return (1 - self.temporal_weight) * visual_sim + self.temporal_weight * temporal_sim

    def _find_representative_frames(self, frames: List[VideoFrame]) -> int:
        """Find the most representative frame in a cluster"""
        features = np.array([frame.features for frame in frames])
        center = features.mean(axis=0)
        similarities = cosine_similarity([center], features)[0]
        return np.argmax(similarities)

    def build_tree(self, frames: List[VideoFrame]) -> VideoTree:
        """Build hierarchical tree structure from video frames"""
        try:
            n_frames = len(frames)
            if n_frames == 0:
                raise ValueError("No frames provided for tree building")

            # Step 1: Hierarchical clustering
            similarity_matrix = self._compute_combined_similarity(frames)
            clustering = AgglomerativeClustering(
                n_clusters=None,
                affinity="precomputed",
                linkage="average",
                distance_threshold=1 - self.similarity_threshold
            )
            clusters = clustering.fit_predict(1 - similarity_matrix)

            # Step 2: Create tree structure
            nodes = {}
            cluster_to_node = {}
            next_node_id = 0

            # Create leaf nodes (each frame is a leaf)
            for frame_idx, cluster_id in enumerate(clusters):
                if cluster_id not in cluster_to_node:
                    node_id = next_node_id
                    next_node_id += 1
                    cluster_to_node[cluster_id] = node_id
                    nodes[node_id] = VideoNode(
                        id=node_id,
                        frame_indices=[frame_idx],
                        children=[],
                        level=0
                    )
                else:
                    node_id = cluster_to_node[cluster_id]
                    nodes[node_id].frame_indices.append(frame_idx)

            # Step 3: Build hierarchy by merging similar clusters
            current_level = 0
            while len(nodes) > 1:
                # Compute cluster similarities (average linkage)
                cluster_ids = sorted(cluster_to_node.values())
                cluster_sim = np.zeros((len(cluster_ids), len(cluster_ids)))

                for i, node_id_i in enumerate(cluster_ids):
                    frames_i = nodes[node_id_i].frame_indices
                    for j, node_id_j in enumerate(cluster_ids[i + 1:], i + 1):
                        frames_j = nodes[node_id_j].frame_indices

                        # Compute average similarity between clusters
                        total_sim = 0
                        count = 0
                        for idx_i in frames_i:
                            for idx_j in frames_j:
                                total_sim += similarity_matrix[idx_i, idx_j]
                                count += 1
                        if count > 0:
                            cluster_sim[i, j] = cluster_sim[j, i] = total_sim / count

                # Find most similar clusters to merge
                merged = set()
                new_nodes = {}

                for i, node_id_i in enumerate(cluster_ids):
                    if node_id_i in merged:
                        continue

                    # Find most similar neighbor not already merged
                    best_sim = -1
                    best_j = -1

                    for j, node_id_j in enumerate(cluster_ids):
                        if (i != j and
                                node_id_j not in merged and
                                cluster_sim[i, j] > best_sim and
                                cluster_sim[i, j] >= self.similarity_threshold):
                            best_sim = cluster_sim[i, j]
                            best_j = j

                    if best_j != -1:
                        # Merge nodes
                        node_id_j = cluster_ids[best_j]
                        parent_id = next_node_id
                        next_node_id += 1

                        # Create new parent node
                        new_nodes[parent_id] = VideoNode(
                            id=parent_id,
                            frame_indices=nodes[node_id_i].frame_indices + nodes[node_id_j].frame_indices,
                            children=[node_id_i, node_id_j],
                            level=current_level + 1
                        )

                        # Update children's parent
                        nodes[node_id_i].parent = parent_id
                        nodes[node_id_j].parent = parent_id

                        merged.add(node_id_i)
                        merged.add(node_id_j)
                    else:
                        # No merge, promote to next level
                        new_nodes[node_id_i] = VideoNode(
                            id=node_id_i,
                            frame_indices=nodes[node_id_i].frame_indices,
                            children=nodes[node_id_i].children,
                            parent=None,
                            level=current_level + 1
                        )

                nodes = new_nodes
                current_level += 1

            # Step 4: Set representative frames
            for node in nodes.values():
                if len(node.frame_indices) > 0:
                    node.representative_frame_idx = self._find_representative_frames(
                        [frames[i] for i in node.frame_indices]
                    )

            return VideoTree(nodes=nodes, frames=frames)

        except Exception as e:
            logger.error(f"Tree building failed: {str(e)}")
            raise