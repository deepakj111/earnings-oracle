from unittest.mock import MagicMock, patch

import generation
from generation import Generator, _get_generator


def test_get_generator_singleton():
    # Reset singleton if exists
    generation._generator = None
    gen1 = _get_generator()
    gen2 = _get_generator()
    assert gen1 is gen2
    assert isinstance(gen1, Generator)


@patch("generation._get_generator")
def test_generate_shortcut(mock_get):
    mock_generator = MagicMock()
    mock_get.return_value = mock_generator

    retrieval_result = MagicMock()

    generation.generate("test question", retrieval_result)

    mock_generator.generate.assert_called_once_with("test question", retrieval_result)
