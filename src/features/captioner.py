from __future__ import annotations
from typing import List
from ..config import settings

def generate_captions(image_paths: List[str]) -> list[str]:
    if not settings.CAPTIONING_ENABLED:
        return ["" for _ in image_paths]

    try:
        import torch
        from PIL import Image
        from transformers import BlipProcessor, BlipForConditionalGeneration
    except Exception as e:
        return ["" for _ in image_paths]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = BlipProcessor.from_pretrained(settings.BLIP_MODEL)
    model = BlipForConditionalGeneration.from_pretrained(settings.BLIP_MODEL).to(device).eval()

    caps = []
    with torch.no_grad():
        for p in image_paths:
            raw = Image.open(p).convert("RGB")
            inputs = processor(raw, return_tensors="pt").to(device)
            out = model.generate(**inputs, max_new_tokens=30)
            text = processor.decode(out[0], skip_special_tokens=True)
            caps.append(text)
    return caps
