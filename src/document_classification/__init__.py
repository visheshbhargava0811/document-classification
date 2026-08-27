"""Document scanning + classification pipeline.

Public API:
    from document_classification import process_path, process_image, ScanResult
"""
from __future__ import annotations

from .pipeline import ScanResult, load_image, process_image, process_path

__all__ = ["ScanResult", "load_image", "process_image", "process_path", "main"]


def main() -> None:
    """Console entry point (delegates to the CLI)."""
    from .cli import main as _cli_main

    _cli_main()
