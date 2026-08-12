"""Public API for WBS SAM2 inference."""

__version__ = "0.1.0"

from .config import InferenceConfig, load_config
from .pipeline import InferenceResult, WBSSegmenter

__all__ = ["InferenceConfig", "InferenceResult", "WBSSegmenter", "load_config"]
