# BGF-YOLO based on Ultralytics YOLOv8x 8.0.109 object detection model with same license, AGPL-3.0 license

import os
import sys

# Add the project root to the Python path
project_root = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
sys.path.insert(0, project_root)

__all__ = ["detect"]


def __getattr__(name):
    """Lazy import for detect module to avoid circular imports."""
    if name == "detect":
        import importlib

        detect_module = importlib.import_module("yolo.bgf.detect")
        globals()[name] = detect_module
        return detect_module
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
