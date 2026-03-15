"""
conftest.py — pytest configuration

Sets environment variables before tests run.
Ensures model path is correct regardless of where pytest is called from.
"""

import os
import pytest
from pathlib import Path

# Project root — two levels up from tests/
PROJECT_ROOT = Path(__file__).parent.parent


def pytest_configure(config):
    """Set environment variables before any tests run."""
    model_path = PROJECT_ROOT / 'model' / 'model.joblib'
    os.environ['MODEL_PATH'] = str(model_path)