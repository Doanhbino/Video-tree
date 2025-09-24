import os
from dataclasses import dataclass

try:
    from dotenv import load_dotenv
    load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "..", ".env"))
except Exception:
    pass

@dataclass
class Settings:
    # Paths
    DATA_DIR: str = os.path.join(os.path.dirname(os.path.dirname(__file__)), "..", "data")
    FRAMES_DIRNAME: str = "frames"
    EMB_DIRNAME: str = "embeddings"
    TREE_JSON: str = "tree.json"

    # Video processing
    TARGET_FPS: float = 0.5
    MAX_FRAMES: int | None = None
    FRAME_SIZE: int = 224

    # CLIP
    CLIP_MODEL: str = "openai/clip-vit-base-patch32"
    CLIP_DEVICE: str = "cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") else "cpu"
    CLIP_BATCH: int = 32

    # Tree
    SIMILARITY_THRESHOLD: float = 0.88
    MAX_CHILDREN_PER_NODE: int = 8

    # Captioning (offline BLIP)
    CAPTIONING_ENABLED: bool = True
    BLIP_MODEL: str = "Salesforce/blip-image-captioning-base"

    # LLM
    USE_OPENAI: bool = True
    OPENAI_MODEL: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    OPENAI_API_KEY: str | None = os.getenv("OPENAI_API_KEY")

settings = Settings()

os.makedirs(settings.DATA_DIR, exist_ok=True)
