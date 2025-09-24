from __future__ import annotations
import torch
import numpy as np
from PIL import Image
from typing import List
from transformers import CLIPProcessor, CLIPModel
from ..config import settings

class CLIPFeatureExtractor:
    def __init__(self, model_name: str | None = None, device: str | None = None, batch_size: int | None = None):
        self.model_name = model_name or settings.CLIP_MODEL
        self.device = device or settings.CLIP_DEVICE
        self.batch_size = batch_size or settings.CLIP_BATCH
        self.model = CLIPModel.from_pretrained(self.model_name)
        self.processor = CLIPProcessor.from_pretrained(self.model_name)
        self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def encode_images(self, image_paths: List[str]) -> np.ndarray:
        embs = []
        for i in range(0, len(image_paths), self.batch_size):
            batch_paths = image_paths[i:i+self.batch_size]
            images = [Image.open(p).convert("RGB") for p in batch_paths]
            inputs = self.processor(images=images, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            outputs = self.model.get_image_features(**inputs)
            batch_embs = outputs / outputs.norm(dim=-1, keepdim=True)
            embs.append(batch_embs.cpu().numpy())
        return np.concatenate(embs, axis=0)

    @torch.no_grad()
    def encode_text(self, texts: List[str]) -> np.ndarray:
        inputs = self.processor(text=texts, return_tensors="pt", padding=True, truncation=True).to(self.device)
        outputs = self.model.get_text_features(**inputs)
        outputs = outputs / outputs.norm(dim=-1, keepdim=True)
        return outputs.cpu().numpy()
