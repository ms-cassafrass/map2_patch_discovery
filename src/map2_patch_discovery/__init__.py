"""
MAP2 ROI-driven patch extraction and dataset scaffolding.
"""

from .config import load_dataset_config
from .pipeline import run_patch_extraction

__all__ = ["load_dataset_config", "run_patch_extraction"]
