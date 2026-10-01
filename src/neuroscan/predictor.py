"""High-level Diagnostic Inference Pipeline combining Preprocessing, VGG-16, and Grad-CAM."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np
from PIL import Image

from neuroscan.classifier import VGG16BrainClassifier
from neuroscan.gradcam import overlay_heatmap_on_image
from neuroscan.models import DiagnosticLabel, DiagnosticReport
from neuroscan.preprocess import preprocess_mri


class NeuroPredictor:
    """End-to-end diagnostic inference engine for brain MRI scans."""

    def __init__(
        self,
        weights_path: Optional[Path] = None,
        threshold: float = 0.50,
    ) -> None:
        self.threshold = threshold
        self.classifier = VGG16BrainClassifier(weights_path=weights_path)

    def predict(
        self,
        image_input: Union[str, Path, Image.Image, np.ndarray],
        output_dir: Optional[Path] = None,
        save_visualizations: bool = False,
    ) -> DiagnosticReport:
        """
        Executes complete diagnostic pipeline:
        1. Extreme-point skull contour cropping
        2. VGG-16 Forward inference
        3. Grad-CAM diagnostic heatmap extraction
        4. Visualization fusion & report generation
        """
        # Resolve path if file
        path_ref = Path(image_input) if isinstance(image_input, (str, Path)) else Path("in_memory_scan.jpg")

        # 1. Preprocess & crop
        tensor_224, cropped_rgb, bbox = preprocess_mri(
            image_input,
            target_size=(224, 224),
            apply_crop=True,
        )

        # 2. Classifier inference & Grad-CAM
        tumor_prob, heatmap_224 = self.classifier.predict_with_gradcam(tensor_224)

        # 3. Diagnostic determination
        is_tumor = tumor_prob >= self.threshold
        label = DiagnosticLabel.TUMOR_DETECTED if is_tumor else DiagnosticLabel.HEALTHY
        confidence = tumor_prob if is_tumor else (1.0 - tumor_prob)

        # 4. Generate visual overlay
        overlay_rgb = overlay_heatmap_on_image(cropped_rgb, heatmap_224, alpha=0.45)

        cropped_path = None
        gradcam_path = None
        overlay_path = None

        if save_visualizations and output_dir:
            output_dir.mkdir(parents=True, exist_ok=True)
            stem = path_ref.stem

            cropped_path = output_dir / f"{stem}_cropped.png"
            gradcam_path = output_dir / f"{stem}_gradcam.png"
            overlay_path = output_dir / f"{stem}_overlay.png"

            Image.fromarray(cropped_rgb).save(cropped_path)
            Image.fromarray((heatmap_224 * 255).astype(np.uint8)).save(gradcam_path)
            Image.fromarray(overlay_rgb).save(overlay_path)

        notes = (
            f"Focal hyper-intensity consistent with neoplastic lesion (p={tumor_prob:.2%})"
            if is_tumor
            else f"No anomalous focal hyper-intensity detected (p={tumor_prob:.2%})"
        )

        return DiagnosticReport(
            image_path=path_ref,
            label=label,
            confidence=float(confidence),
            raw_score=float(tumor_prob),
            bbox=bbox,
            cropped_image_path=cropped_path,
            gradcam_image_path=gradcam_path,
            overlay_image_path=overlay_path,
            notes=notes,
        )
