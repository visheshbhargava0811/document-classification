"""Root entry point for Streamlit Community Cloud.

Cloud runs from the repo root without installing the package, so put ``src`` on
the path, then import the app module (importing it runs the Streamlit script).
Point the Streamlit Cloud "Main file path" at this file.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

import document_classification.app  # noqa: E402,F401
