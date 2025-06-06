import pytest
from unittest.mock import patch, MagicMock
from app.services import rag_pipeline
from app.core import config


def test_create_prompt_with_context_chunks():
    query = "What is the meaning of life?"
    chunks = [
        {"text": "Context chunk 1 about life.", "metadata": {}},
        {"text": "Context chunk 2 mentioning the universe.", "metadata": {}}
    ]
    prompt = rag_pipeline.create_prompt_with_context(query, chunks)
    assert query in prompt
    assert "Context chunk 1 about life." in prompt
    assert "Context chunk 2 mentioning the universe." in prompt
    assert "Answer the question based *only* on the following context." in prompt


def test_create_prompt_with_no_context_chunks():
    query = "What is AI?"
    prompt = rag_pipeline.create_prompt_with_context(query, [])
    assert query in prompt
    assert "no specific context from documents was found" in prompt


# Mocking the llm_client for get_llm_response
@patch('app.services.rag_pipeline.llm_client')  # Target the llm_client where it's USED
def test_get_llm_response_success(mock_llm_client):
    # Configure the mock client's behavior
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = "This is a mock LLM answer."
    mock_llm_client.chat.completions.create.return_value = mock_response

    prompt = "Test prompt"
    answer = rag_pipeline.get_llm_response(prompt)

    mock_llm_client.chat.completions.create.assert_called_once()  # Verify it was called
    assert answer == "This is a mock LLM answer."


@patch('app.services.rag_pipeline.llm_client')
def test_get_llm_response_api_error(mock_llm_client):
    # Simulate an API error
    mock_llm_client.chat.completions.create.side_effect = Exception("LLM API Error")

    prompt = "Test prompt for error"
    answer = rag_pipeline.get_llm_response(prompt)

    assert answer is None


@patch('app.services.rag_pipeline.llm_client', None)  # Simulate client not initialized
def test_get_llm_response_client_not_initialized():
    original_client = rag_pipeline.llm_client
    rag_pipeline.llm_client = None  # Directly set to None for this test

    answer = rag_pipeline.get_llm_response("A prompt")
    assert answer is None

    rag_pipeline.llm_client = original_client  # Restore for other tests
