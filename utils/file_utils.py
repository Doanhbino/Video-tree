import os
import tempfile
from typing import Tuple


def save_uploaded_file(file_bytes: bytes, extension: str = "mp4") -> Tuple[str, str]:
    """Save uploaded file to temporary location"""
    temp_dir = tempfile.mkdtemp()
    file_path = os.path.join(temp_dir, f"upload.{extension}")

    with open(file_path, "wb") as f:
        f.write(file_bytes)

    return file_path, temp_dir


def cleanup_temp_files(temp_dir: str):
    """Clean up temporary files"""
    try:
        for filename in os.listdir(temp_dir):
            file_path = os.path.join(temp_dir, filename)
            if os.path.isfile(file_path):
                os.unlink(file_path)
        os.rmdir(temp_dir)
    except Exception as e:
        pass  # Silently fail cleanup