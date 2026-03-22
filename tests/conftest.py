import pytest
from hydra.core.global_hydra import GlobalHydra


@pytest.fixture(autouse=True)
def clear_hydra():
    """Clear Hydra global state before each test."""
    GlobalHydra.instance().clear()
    yield
    GlobalHydra.instance().clear()
