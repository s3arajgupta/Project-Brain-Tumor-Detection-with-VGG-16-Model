"""Data representations and diagnostic predictions for NeuroScan."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional, Tuple


class DiagnosticLabel(str, Enum):
    """Clinical binary classification outcomes."""
    TUMOR_DETECTED = "Tumor Detected"
    HEALTHY = "Healthy / No Tumor"


@dataclass
class BoundingBox:
    """Extreme point coordinates defining the cropped brain parenchyma."""
    min_x: int
    max_x: int
    min_y: int
    max_y: int

    @property
    def width(self) -> int:
        return max(0, self.max_x - self.min_x)

    @property
    def height(self) -> int:
        return max(0, self.max_y - self.min_y)


@dataclass
class DiagnosticReport:
    """Complete diagnostic assessment for a brain MRI scan."""
    image_path: Path
    label: DiagnosticLabel
    confidence: float
    raw_score: float
    bbox: Optional[BoundingBox] = None
    cropped_image_path: Optional[Path] = None
    gradcam_image_path: Optional[Path] = None
    overlay_image_path: Optional[Path] = None
    notes: str = ""

    @property
    def is_positive(self) -> bool:
        return self.label == DiagnosticLabel.TUMOR_DETECTED
