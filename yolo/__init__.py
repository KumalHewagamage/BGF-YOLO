# BGF-YOLO based on Ultralytics YOLOv8x 8.0.109 object detection model with same license, AGPL-3.0 license

__all__ = ["bgf"]  # tuple or list


def __getattr__(name):
    """Lazy import for bgf module to avoid circular imports."""
    if name == "bgf":
        import importlib

        bgf_module = importlib.import_module("yolo.bgf")
        globals()[name] = bgf_module
        return bgf_module
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
