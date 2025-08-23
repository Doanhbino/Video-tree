import unittest
import numpy as np
from unittest.mock import patch, MagicMock
from core.video_processor import VideoProcessor
from core.models import VideoFrame


class TestVideoProcessor(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.processor = VideoProcessor()

    @patch('decord.VideoReader')
    def test_extract_frames(self, mock_videoreader):
        # Mock video reader
        mock_reader = MagicMock()
        mock_reader.get_avg_fps.return_value = 30
        mock_reader.__len__.return_value = 100
        mock_reader.__getitem__.return_value.asnumpy.return_value = np.zeros((480, 640, 3), dtype=np.uint8)
        mock_videoreader.return_value = mock_reader

        frames = self.processor.extract_frames("dummy.mp4", frame_rate=1)
        self.assertEqual(len(frames), 30)  # 30 seconds at 1 fps

    @patch('transformers.CLIPProcessor.from_pretrained')
    @patch('transformers.CLIPModel.from_pretrained')
    def test_extract_features(self, mock_model, mock_processor):
        # Mock CLIP model and processor
        mock_model.return_value.get_image_features.return_value = torch.randn(2, 512)
        mock_processor.return_value.return_values = {'pixel_values': torch.randn(2, 3, 224, 224)}

        frames = [np.zeros((480, 640, 3), dtype=np.uint8), np.zeros((480, 640, 3), dtype=np.uint8)]
        features = self.processor.extract_features(frames)
        self.assertEqual(features.shape, (2, 512))


if __name__ == '__main__':
    unittest.main()