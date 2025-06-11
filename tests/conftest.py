import pytest
import tempfile
from typing import Generator

try:
    import pyarrow

    assert pyarrow is not None

    PYARROW_AVAILABLE = True
except ImportError:
    PYARROW_AVAILABLE = False


@pytest.fixture
def temp_checkpoint_dir() -> Generator[str, None, None]:
    """Provide a temporary directory for checkpoint tests."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield temp_dir


@pytest.fixture
def skip_without_pyarrow():
    """Skip test if PyArrow is not available."""
    if not PYARROW_AVAILABLE:
        pytest.skip("PyArrow not available")


# Configure pytest to show full diff on assertion failures
def pytest_configure(config):
    """Configure pytest settings."""
    # Ensure we get detailed assertion output
    config.option.tb = "short"
