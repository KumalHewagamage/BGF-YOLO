# Ultralytics YOLO 🚀, AGPL-3.0 license

import os
import sys

# Add the project root to the Python path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from yolo.utils import emojis


class HUBModelError(Exception):

    def __init__(self, message='Model not found. Please check model URL and try again.'):
        """Create an exception for when a model is not found."""
        super().__init__(emojis(message))
