"""."""

import os

from pathlib import Path

import pytest


@pytest.fixture
def files_dir() -> Path:
    """."""
    return Path(__file__).parent / 'files'


@pytest.fixture(autouse=True)
def chdir_tmp(tmp_path) -> Path:
    """."""
    os.chdir(tmp_path)
    return tmp_path
