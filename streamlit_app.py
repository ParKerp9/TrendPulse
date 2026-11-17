"""Root-level entry point for TrendPulse Streamlit Application.

Usage: streamlit run streamlit_app.py
"""

import sys
from pathlib import Path

# Ensure root directory and app directory are present in sys.path
_ROOT_DIR = str(Path(__file__).resolve().parent)
_APP_DIR = str(Path(__file__).resolve().parent / "app")

if _APP_DIR not in sys.path:
    sys.path.insert(0, _APP_DIR)
if _ROOT_DIR not in sys.path:
    sys.path.insert(0, _ROOT_DIR)

# Import main from the app package — use importlib to avoid the "app" name collision
import importlib.util

_spec = importlib.util.spec_from_file_location("_trendpulse_app", _APP_DIR + "/app.py")
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

if __name__ == "__main__":
    _mod.main()
