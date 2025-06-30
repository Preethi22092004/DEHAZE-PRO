# Models module for Perfect Dehazing
# Import from the main models.py file
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

try:
    from models import db, ProcessedImage, ProcessedVideo
    __all__ = ['db', 'ProcessedImage', 'ProcessedVideo']
except ImportError:
    # Fallback if models.py doesn't exist
    pass
