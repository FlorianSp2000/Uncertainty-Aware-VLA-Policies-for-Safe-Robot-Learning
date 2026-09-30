"""Fixtures for the local regression tests.

These tests assert over analysis artifacts (results JSONs), which are gitignored
and only exist on a machine that ran the pipeline. Every fixture that needs one
skips loudly when it is absent rather than failing.
"""

import json
from pathlib import Path

import pytest


def pytest_addoption(parser):
    parser.addoption("--results-dir", action="store", default="src/results",
                     help="directory holding the analysis JSONs under test")


@pytest.fixture(scope="session")
def results_dir(pytestconfig):
    return Path(pytestconfig.getoption("--results-dir"))


@pytest.fixture(scope="session")
def load_results(results_dir):
    def _load(name):
        path = results_dir / name
        if not path.exists():
            pytest.skip(f"{path} absent - produce it with the analysis script first")
        return json.loads(path.read_text(encoding="utf-8"))
    return _load


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: minutes, or loads a >0.5 GB dump (deselect with -m 'not slow')")
