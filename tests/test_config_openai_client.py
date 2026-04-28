# tests/test_config_openai_client.py
from unittest.mock import MagicMock, patch

import pytest

from config.openai_client import get_async_openai_client, get_openai_client, reset_clients


def _mock_openai_response(content: str) -> MagicMock:
    mock_choice = MagicMock()
    mock_choice.message.content = content
    mock_usage = MagicMock()
    mock_usage.prompt_tokens = 50
    mock_usage.completion_tokens = 30
    mock_response = MagicMock()
    mock_response.choices = [mock_choice]
    mock_response.usage = mock_usage
    return mock_response


def _mock_openai_client(content: str = "ok") -> MagicMock:
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = _mock_openai_response(content)
    return mock_client


class TestConfigOpenAIClient:
    def setup_method(self):
        reset_clients()

    def test_missing_api_key_raises_oserror(self):
        mock_settings = MagicMock()
        mock_settings.infra.openai_api_key = ""
        with patch("config.openai_client.settings", mock_settings):
            with pytest.raises(OSError, match="OPENAI_API_KEY"):
                get_openai_client()

    def test_sync_client_singleton_reused(self):
        mock_client_instance = _mock_openai_client()
        mock_settings = MagicMock()
        mock_settings.infra.openai_api_key = "sk-test"

        with patch("config.openai_client.settings", mock_settings):
            with patch(
                "config.openai_client.OpenAI", return_value=mock_client_instance
            ) as mock_class:
                c1 = get_openai_client()
                c2 = get_openai_client()

        assert c1 is c2
        assert mock_class.call_count == 1

    def test_async_client_singleton_reused(self):
        mock_client_instance = MagicMock()
        mock_settings = MagicMock()
        mock_settings.infra.openai_api_key = "sk-test"

        with patch("config.openai_client.settings", mock_settings):
            with patch(
                "config.openai_client.AsyncOpenAI", return_value=mock_client_instance
            ) as mock_class:
                c1 = get_async_openai_client()
                c2 = get_async_openai_client()

        assert c1 is c2
        assert mock_class.call_count == 1
