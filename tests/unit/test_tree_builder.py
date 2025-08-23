import unittest
import numpy as np
from core.tree_builder import VideoTreeBuilder
from core.models import VideoFrame


class TestVideoTreeBuilder(unittest.TestCase):
    def setUp(self):
        self.builder = VideoTreeBuilder(similarity_threshold=0.5)

    def test_build_tree(self):
        # Create dummy frames with features
        frames = [
            VideoFrame(
                index=0,
                timestamp=0,
                image=np.zeros((100, 100, 3), dtype=np.uint8),
                features=np.array([1.0, 0.9, 0.8])
            ),
            VideoFrame(
                index=1,
                timestamp=1,
                image=np.zeros((100, 100, 3), dtype=np.uint8),
                features=np.array([0.9, 1.0, 0.7])
            ),
            VideoFrame(
                index=2,
                timestamp=2,
                image=np.zeros((100, 100, 3), dtype=np.uint8),
                features=np.array([0.8, 0.7, 1.0])
            )
        ]

        tree = self.builder.build_tree(frames)
        self.assertGreater(len(tree.nodes), 0)
        self.assertEqual(len(tree.frames), 3)


if __name__ == '__main__':
    unittest.main()