import os
from dotenv import load_dotenv
from typing import Optional

load_dotenv()


class Settings:
    # Video Processing
    DEFAULT_FRAME_RATE: float = 1.0
    MAX_FRAME_WIDTH: int = 640

    # Tree Building
    DEFAULT_SIMILARITY_THRESHOLD: float = 0.75
    DEFAULT_MAX_CHILDREN: int = 4

    # LLM
    LLM_MODEL_NAME: str = "gpt-4-1106-preview"
    LLM_TEMPERATURE: float = 0.3

    # Paths
    TEMP_DIR: str = "temp"

    @property
    def OPENAI_API_KEY(self) -> Optional[str]:
        return os.getenv("OPENAI_API_KEY")

    @property
    def CACHE_DIR(self) -> str:
        cache_dir = os.getenv("CACHE_DIR", "cache")
        os.makedirs(cache_dir, exist_ok=True)
        return cache_dir


settings = Settings()