import sys
import os

# Add ML_side root to path so tests can import local packages such as depth.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Add deployment folder to path so test_api.py can import api.py
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "deployment")))
