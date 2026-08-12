from pathlib import Path

import pytest

from wbs_inference.config import load_config

REPOSITORY = Path(__file__).resolve().parents[1]


@pytest.fixture
def default_config():
    return load_config(REPOSITORY / "configs" / "default.json")
