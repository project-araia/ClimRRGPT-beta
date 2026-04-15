import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


@pytest.fixture(autouse=True)
def mock_streamlit(mocker):
    """Automatically mock streamlit for all tests."""
    mock_st = mocker.patch("streamlit.empty")
    mocker.patch("streamlit.chat_message")
    mocker.patch("streamlit.write_stream")
    mocker.patch("streamlit.session_state", {})
    return mock_st


@pytest.fixture
def mock_llm_response(mocker):
    """Fixture to mock LLM responses."""

    def _mock(response_text):
        mock_get_response = MagicMock(return_value=response_text)
        mocker.patch("src.llms.get_default_llm", return_value=mock_get_response)
        return mock_get_response

    return _mock


@pytest.fixture
def temp_config(tmp_path, mocker):
    """Fixture to create a temporary config file for testing."""
    config_content = """
literature:
  data_dir: "fake_data"
  dense_index: "dense.faiss"
  sparse_index: "sparse.pkl"
  manifest: "manifest.json"
"""
    config_file = tmp_path / "config.yml"
    config_file.write_text(config_content)
    mocker.patch("src.utils.REPO_ROOT", tmp_path)
    return config_file
