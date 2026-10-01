#!/usr/bin/env python3
"""
CLI entrypoint for NeuroScan Brain Tumor AI.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add src/ to path so this script can be executed directly
sys.path.insert(0, str(Path(__file__).parent / "src"))

from neuroscan.cli import app

if __name__ == "__main__":
    app()
