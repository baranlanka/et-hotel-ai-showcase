"""
This file contains shared fixtures and configuration for the test suite.
"""
import os
import sys
from pathlib import Path

import pytest
import pandas as pd
from unittest.mock import Mock

# NOTE: The showcase package (llm_content_generation.core.config) has no
# required environment variables, so the large app-config env stub the original
# suite needed is intentionally omitted here. The trimmed slice imports none of
# the app.* settings that previously forced those placeholders.

# Add the project root to the Python path to allow for absolute imports
# The conftest.py is in llm_content_generation/tests, so we need to go up two levels
project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))


@pytest.fixture
def sample_review_data():
    """Sample review data for testing."""
    return [
        {
            "text": "Great hotel with amazing amenities",
            "rating": 5,
            "date": "2024-01-15",
            "source": "demo_ota",
            "language": "en"
        },
        {
            "text": "Nice location but rooms could be cleaner",
            "rating": 3,
            "date": "2024-01-14",
            "source": "tripadvisor",
            "language": "en"
        }
    ]

@pytest.fixture
def sample_dataframe():
    """Sample DataFrame for testing."""
    return pd.DataFrame({
        "text": ["Great hotel", "Nice location"],
        "rating": [5, 3],
        "date": ["2024-01-15", "2024-01-14"],
        "source": ["demo_ota", "tripadvisor"],
        "language": ["en", "en"]
    })

@pytest.fixture
def mock_llm():
    """Mock LLM for testing."""
    mock = Mock()
    mock.invoke.return_value = {
        "content": '{"sentiment": "positive", "key_points": ["amenities", "service"]}'
    }
    return mock

@pytest.fixture
def mock_storage():
    """Mock storage service for testing."""
    mock = Mock()
    mock.exists.return_value = True
    mock.save_dataframe.return_value = "test/path.parquet"
    mock.save_json.return_value = "test/path.json"
    mock.load_dataframe.return_value = pd.DataFrame({"test": [1, 2, 3]})
    mock.read_json_file.return_value = {"test": "data"}
    return mock
