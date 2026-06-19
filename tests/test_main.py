"""
Tests for the Mistral FastAPI application.
"""
import pytest
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient

import warnings

warnings.filterwarnings("ignore")

# --- Fixtures ---


@pytest.fixture
def mock_mistral_client():
    """Patch the Mistral client so no real API calls are made."""
    with patch.dict("os.environ", {"MISTRAL_API_KEY": "test-key"}):
        with patch("mistralai.Mistral") as MockMistral:
            mock_client = MagicMock()
            MockMistral.return_value = mock_client
            yield mock_client


@pytest.fixture
def test_client(mock_mistral_client):
    """Create a FastAPI TestClient with the mocked Mistral client injected."""
    # Import app after env var and mock are set up
    import importlib
    import reranker.main as main

    importlib.reload(main)  # reload to pick up patched client
    main.client = mock_mistral_client

    return TestClient(main.app)


# --- Tests ---


class TestRootEndpoint:
    def test_read_root_status_code(self, test_client):
        response = test_client.get("/")
        assert response.status_code == 200

    def test_read_root_response_body(self, test_client):
        response = test_client.get("/")
        assert response.json() == {"Hello": "World"}


class TestChatEndpoint:
    def _mock_response(self, mock_client, content: str):
        """Helper to configure what the mocked Mistral client returns."""
        mock_choice = MagicMock()
        mock_choice.message.content = content
        mock_completion = MagicMock()
        mock_completion.choices = [mock_choice]
        mock_client.chat.complete.return_value = mock_completion

    def test_chat_returns_200(self, test_client, mock_mistral_client):
        self._mock_response(mock_mistral_client, "Paris is the capital of France.")
        response = test_client.get("/chat/What is the capital of France?")
        assert response.status_code == 200

    def test_chat_returns_response_key(self, test_client, mock_mistral_client):
        self._mock_response(mock_mistral_client, "Paris is the capital of France.")
        response = test_client.get("/chat/What is the capital of France?")
        assert "response" in response.json()

    def test_chat_returns_model_content(self, test_client, mock_mistral_client):
        expected = "Paris is the capital of France."
        self._mock_response(mock_mistral_client, expected)
        response = test_client.get("/chat/What is the capital of France?")
        assert response.json()["response"] == expected

    def test_chat_calls_mistral_with_correct_model(
        self, test_client, mock_mistral_client
    ):
        self._mock_response(mock_mistral_client, "42")
        test_client.get("/chat/What is the answer?")
        call_kwargs = mock_mistral_client.chat.complete.call_args
        assert call_kwargs.kwargs["model"] == "mistral-medium-3-5"

    def test_chat_empty_response_from_model(self, test_client, mock_mistral_client):
        self._mock_response(mock_mistral_client, "")
        response = test_client.get("/chat/Silence?")
        assert response.status_code == 200
        assert response.json()["response"] == ""
