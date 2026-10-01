import numpy as np
import pytest

from neuroscan.models import DiagnosticLabel, DiagnosticReport
from neuroscan.predictor import NeuroPredictor


def test_neuropredictor_synthetic_inference():
    predictor = NeuroPredictor()

    # Synthetic image tensor
    img = np.full((120, 120, 3), 150, dtype=np.uint8)
    report = predictor.predict(img)

    assert isinstance(report, DiagnosticReport)
    assert report.label in (DiagnosticLabel.TUMOR_DETECTED, DiagnosticLabel.HEALTHY)
    assert 0.0 <= report.confidence <= 1.0
    assert 0.0 <= report.raw_score <= 1.0
    assert report.bbox is not None
    assert len(report.notes) > 0
