import hashlib
import re
from pathlib import Path


def file_hash(path: str) -> str:
    """MD5 hash of a file — used for deduplication."""
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def sanitise_filename(name: str) -> str:
    """Remove unsafe characters from filenames."""
    return re.sub(r"[^\w\-_\. ]", "_", name)


def truncate_text(text: str, max_chars: int = 300) -> str:
    return text[:max_chars] + "..." if len(text) > max_chars else text


def detect_file_type(path: str) -> str:
    ext = Path(path).suffix.lower()
    mapping = {".pdf": "pdf", ".png": "image", ".jpg": "image",
               ".jpeg": "image", ".tiff": "image", ".bmp": "image"}
    return mapping.get(ext, "unknown")
