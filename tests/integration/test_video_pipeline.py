import unittest
import numpy as np
from core.video_processor import VideoProcessor
from core.tree_builder import VideoTreeBuilder
from core.models import VideoFrame


class TestVideoPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.processor = VideoProcessor()
        cls.builder = VideoTreeBuilder()

    def test_full_pipeline(self):
        # Create dummy video frames
        dummy_frames = [np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8) for _ in range(5)]

        # Test feature extraction
        features = self.processor.extract_features(dummy_frames)
        self.assertEqual(features.shape[0], 5)

        # Test tree building
        frames = [
            VideoFrame(index=i, timestamp=i, image=dummy_frames[i], features=features[i])
            for i in range(5)
        ]
        tree = self.builder.build_tree(frames)
        self.assertEqual(len(tree.nodes), 3)  # Should have some hierarchy


if __name__ == '__main__':
    unittest.main()