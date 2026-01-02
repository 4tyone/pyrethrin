"""Tests for OpenAI shield."""

import pytest
from unittest.mock import Mock, patch, MagicMock

from pyrethrin.shields import openai
from pyrethrin import match, Ok, Err
from pyrethrin.decorators import ExhaustiveSignature


class TestOpenAIShieldImports:
    """Test that the shield imports correctly."""

    def test_shield_version(self):
        assert openai.__shield_version__ == "0.1.0"

    def test_openai_version(self):
        # Should get version from actual openai package
        assert openai.__openai_version__ is not None

    def test_exceptions_available(self):
        """All OpenAI exceptions should be re-exported."""
        exceptions = [
            "OpenAIError",
            "APIError",
            "APIStatusError",
            "APIConnectionError",
            "APITimeoutError",
            "BadRequestError",
            "AuthenticationError",
            "PermissionDeniedError",
            "NotFoundError",
            "ConflictError",
            "UnprocessableEntityError",
            "RateLimitError",
            "InternalServerError",
        ]
        for exc in exceptions:
            assert hasattr(openai, exc), f"Missing exception: {exc}"

    def test_types_available(self):
        """Common types should be re-exported."""
        types = [
            "NOT_GIVEN",
            "NotGiven",
            "Timeout",
            "Stream",
            "AsyncStream",
            "BaseModel",
        ]
        for t in types:
            assert hasattr(openai, t), f"Missing type: {t}"

    def test_clients_available(self):
        """Client classes should be available."""
        assert hasattr(openai, "OpenAI")
        assert hasattr(openai, "AsyncOpenAI")
        assert hasattr(openai, "Client")
        assert hasattr(openai, "AsyncClient")
        # Aliases should be the same
        assert openai.Client is openai.OpenAI
        assert openai.AsyncClient is openai.AsyncOpenAI


class TestModuleLevelPassthrough:
    """Test that module-level __getattr__ works for passthrough."""

    def test_can_access_types_submodule(self):
        """Should be able to access openai.types."""
        assert hasattr(openai, "types")

    def test_passthrough_for_unknown_attributes(self):
        """Should passthrough unknown attributes to openai module."""
        # This should not raise - it passes through to openai
        import openai as real_openai

        # Check that we can access things that exist in real openai
        assert hasattr(real_openai, "resources")
        # Our passthrough should work too
        assert openai.resources is real_openai.resources


class TestShieldedChatCompletions:
    """Test chat.completions.create shielding."""

    def test_has_raises_decorator(self):
        """The create method should have @raises applied."""
        from pyrethrin.shields.openai import _ShieldedChatCompletions

        create = _ShieldedChatCompletions.create
        assert hasattr(create, "__pyrethrin_raises__")

        raises_set = create.__pyrethrin_raises__
        assert openai.APIConnectionError in raises_set
        assert openai.APITimeoutError in raises_set
        assert openai.RateLimitError in raises_set
        assert openai.BadRequestError in raises_set
        assert openai.AuthenticationError in raises_set

    def test_create_delegates_to_underlying(self):
        """create() should call the underlying client method."""
        mock_completions = Mock()
        mock_completions.create.return_value = Mock(
            choices=[Mock(message=Mock(content="Hello!"))]
        )

        shielded = openai._ShieldedChatCompletions(mock_completions)
        result = shielded.create(
            model="gpt-4", messages=[{"role": "user", "content": "Hi"}]
        )

        mock_completions.create.assert_called_once()

    def test_passthrough_for_other_attributes(self):
        """Non-shielded attributes should passthrough."""
        mock_completions = Mock()
        mock_completions.some_other_method.return_value = "test"

        shielded = openai._ShieldedChatCompletions(mock_completions)
        result = shielded.some_other_method()

        assert result == "test"
        mock_completions.some_other_method.assert_called_once()


class TestShieldedEmbeddings:
    """Test embeddings.create shielding."""

    def test_has_raises_decorator(self):
        from pyrethrin.shields.openai import _ShieldedEmbeddings

        create = _ShieldedEmbeddings.create
        assert hasattr(create, "__pyrethrin_raises__")

        raises_set = create.__pyrethrin_raises__
        assert openai.RateLimitError in raises_set


class TestShieldedImages:
    """Test images shielding."""

    def test_generate_has_raises(self):
        from pyrethrin.shields.openai import _ShieldedImages

        generate = _ShieldedImages.generate
        assert hasattr(generate, "__pyrethrin_raises__")

        raises_set = generate.__pyrethrin_raises__
        assert openai.ContentFilterFinishReasonError in raises_set


class TestOpenAIClient:
    """Test the main shielded OpenAI client."""

    @patch("openai.OpenAI")
    def test_client_creation(self, mock_openai_class):
        """OpenAI() should create a shielded client."""
        mock_client = Mock()
        mock_openai_class.return_value = mock_client

        client = openai.OpenAI(api_key="test-key")

        mock_openai_class.assert_called_once()
        assert hasattr(client, "chat")
        assert hasattr(client, "embeddings")
        assert hasattr(client, "images")
        assert hasattr(client, "audio")
        assert hasattr(client, "moderations")
        assert hasattr(client, "files")
        assert hasattr(client, "models")
        assert hasattr(client, "fine_tuning")
        assert hasattr(client, "batches")

    @patch("openai.OpenAI")
    def test_chat_completions_accessible(self, mock_openai_class):
        """client.chat.completions should be accessible."""
        mock_client = Mock()
        mock_openai_class.return_value = mock_client

        client = openai.OpenAI(api_key="test-key")

        assert hasattr(client.chat, "completions")
        assert hasattr(client.chat.completions, "create")

    @patch("openai.OpenAI")
    def test_passthrough_for_unknown_attributes(self, mock_openai_class):
        """Unknown attributes should passthrough to underlying client."""
        mock_client = Mock()
        mock_client.some_unknown_attr = "test_value"
        mock_openai_class.return_value = mock_client

        client = openai.OpenAI(api_key="test-key")

        assert client.some_unknown_attr == "test_value"

    @patch("openai.OpenAI")
    def test_context_manager(self, mock_openai_class):
        """Client should support context manager protocol."""
        mock_client = Mock()
        mock_openai_class.return_value = mock_client

        with openai.OpenAI(api_key="test-key") as client:
            assert client is not None

        mock_client.close.assert_called_once()


class TestAsyncOpenAIClient:
    """Test the async shielded OpenAI client."""

    @patch("openai.AsyncOpenAI")
    def test_async_client_creation(self, mock_openai_class):
        """AsyncOpenAI() should create a shielded async client."""
        mock_client = Mock()
        mock_openai_class.return_value = mock_client

        client = openai.AsyncOpenAI(api_key="test-key")

        mock_openai_class.assert_called_once()
        assert hasattr(client, "chat")
        assert hasattr(client, "embeddings")

    @patch("openai.AsyncOpenAI")
    def test_async_chat_completions_accessible(self, mock_openai_class):
        """client.chat.completions should be accessible on async client."""
        mock_client = Mock()
        mock_openai_class.return_value = mock_client

        client = openai.AsyncOpenAI(api_key="test-key")

        assert hasattr(client.chat, "completions")
        assert hasattr(client.chat.completions, "create")


# Get the full set of API exceptions for exhaustive tests
_ALL_API_EXCEPTIONS = {
    openai.APIConnectionError,
    openai.APITimeoutError,
    openai.BadRequestError,
    openai.AuthenticationError,
    openai.PermissionDeniedError,
    openai.NotFoundError,
    openai.ConflictError,
    openai.UnprocessableEntityError,
    openai.RateLimitError,
    openai.InternalServerError,
    openai.APIResponseValidationError,
}


class TestMatchIntegration:
    """Test that the shield works with pyrethrin.match()."""

    def test_match_with_success(self):
        """match() should work with successful API calls."""
        mock_completions = Mock()
        mock_response = Mock()
        mock_response.choices = [Mock(message=Mock(content="Hello!"))]
        mock_completions.create.return_value = mock_response

        shielded = openai._ShieldedChatCompletions(mock_completions)

        # Exhaustive error handling - must handle ALL declared exceptions
        handlers = {Ok: lambda r: r.choices[0].message.content}
        for exc in _ALL_API_EXCEPTIONS:
            handlers[exc] = lambda e, exc=exc: f"error: {exc.__name__}"

        result = match(
            shielded.create, model="gpt-4", messages=[{"role": "user", "content": "Hi"}]
        )(handlers)

        assert result == "Hello!"

    def test_match_with_rate_limit_error(self):
        """match() should handle RateLimitError correctly."""
        mock_completions = Mock()
        mock_response = Mock()
        mock_response.status_code = 429
        mock_response.headers = {}
        mock_completions.create.side_effect = openai.RateLimitError(
            "Rate limit exceeded", response=mock_response, body={}
        )

        shielded = openai._ShieldedChatCompletions(mock_completions)

        # Exhaustive error handling
        handlers = {
            Ok: lambda r: "success",
            openai.RateLimitError: lambda e: "rate limited",
        }
        for exc in _ALL_API_EXCEPTIONS:
            if exc not in handlers:
                handlers[exc] = lambda e, exc=exc: f"error: {exc.__name__}"

        result = match(
            shielded.create, model="gpt-4", messages=[{"role": "user", "content": "Hi"}]
        )(handlers)

        assert result == "rate limited"

    def test_missing_handler_raises_exhaustiveness_error(self):
        """Pyrethrin should raise ExhaustivenessError when handlers are missing."""
        from pyrethrin.exceptions import ExhaustivenessError

        mock_completions = Mock()
        shielded = openai._ShieldedChatCompletions(mock_completions)

        with pytest.raises(ExhaustivenessError) as exc_info:
            match(
                shielded.create,
                model="gpt-4",
                messages=[{"role": "user", "content": "Hi"}],
            )(
                {
                    Ok: lambda r: "success",
                    # Missing handlers for other exceptions!
                }
            )

        # Check that the error message mentions missing exceptions
        assert "Missing handlers" in str(exc_info.value)


class TestDropInReplacement:
    """Test that the shield works as a drop-in replacement."""

    def test_can_import_like_openai(self):
        """Should be able to use shield like original openai package."""
        from pyrethrin.shields import openai as shielded_openai

        # These should all work
        assert shielded_openai.OpenAI is not None
        assert shielded_openai.AsyncOpenAI is not None
        assert shielded_openai.APIError is not None
        assert shielded_openai.types is not None

    def test_file_from_path_available(self):
        """file_from_path helper should be available."""
        assert hasattr(openai, "file_from_path")
        assert callable(openai.file_from_path)
