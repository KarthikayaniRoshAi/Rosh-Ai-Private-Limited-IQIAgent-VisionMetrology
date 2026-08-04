from dataclasses import dataclass
from pathlib import Path


@dataclass
class DrawingPage:
    page_number: int
    image_path: Path
    width: int
    height: int
    dpi: int