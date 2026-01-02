"""Pyrethrin shields for OpenAI SDK.

This module is a drop-in replacement for the `openai` package that adds
explicit exception declarations to API methods via @raises decorators.

Shield version: 0.1.0
OpenAI SDK version: 2.14.0

Usage:
    # Drop-in replacement for: import openai
    from pyrethrin.shields import openai
    from pyrethrin import match, Ok

    client = openai.OpenAI()

    # Shielded methods require exhaustive error handling
    result = match(client.chat.completions.create,
        model="gpt-4",
        messages=[{"role": "user", "content": "Hello!"}]
    )({
        Ok: lambda r: r.choices[0].message.content,
        openai.APIConnectionError: lambda e: "Connection failed",
        openai.APITimeoutError: lambda e: "Request timed out",
        openai.BadRequestError: lambda e: f"Bad request: {e}",
        openai.AuthenticationError: lambda e: "Invalid API key",
        openai.PermissionDeniedError: lambda e: "Permission denied",
        openai.NotFoundError: lambda e: "Not found",
        openai.RateLimitError: lambda e: "Rate limited",
        openai.InternalServerError: lambda e: "Server error",
        openai.APIResponseValidationError: lambda e: "Validation error",
    })

    # Non-shielded methods work exactly like the original SDK
    models = client.models.list()  # No @raises, works normally
"""

from __future__ import annotations

import openai as _openai

__shield_version__ = "0.1.0"
__openai_version__ = getattr(_openai, "__version__", "unknown")

from typing import Any
from collections.abc import Iterable

from pyrethrin.decorators import raises

# =============================================================================
# PASSTHROUGH - Re-export everything from openai
# =============================================================================

# Exceptions
from openai import (
    OpenAIError as OpenAIError,
    APIError as APIError,
    APIStatusError as APIStatusError,
    APIConnectionError as APIConnectionError,
    APITimeoutError as APITimeoutError,
    APIResponseValidationError as APIResponseValidationError,
    BadRequestError as BadRequestError,
    AuthenticationError as AuthenticationError,
    PermissionDeniedError as PermissionDeniedError,
    NotFoundError as NotFoundError,
    ConflictError as ConflictError,
    UnprocessableEntityError as UnprocessableEntityError,
    RateLimitError as RateLimitError,
    InternalServerError as InternalServerError,
    LengthFinishReasonError as LengthFinishReasonError,
    ContentFilterFinishReasonError as ContentFilterFinishReasonError,
    InvalidWebhookSignatureError as InvalidWebhookSignatureError,
)

# Common exports
from openai import (
    NOT_GIVEN as NOT_GIVEN,
    NotGiven as NotGiven,
    Timeout as Timeout,
    RequestOptions as RequestOptions,
    Stream as Stream,
    AsyncStream as AsyncStream,
    BaseModel as BaseModel,
    file_from_path as file_from_path,
)

# Types submodule
from openai import types as types


# =============================================================================
# COMMON EXCEPTION SET - Used by all API methods
# =============================================================================

# These are the exceptions that any OpenAI API call can raise
_API_EXCEPTIONS = (
    APIConnectionError,
    APITimeoutError,
    BadRequestError,
    AuthenticationError,
    PermissionDeniedError,
    NotFoundError,
    ConflictError,
    UnprocessableEntityError,
    RateLimitError,
    InternalServerError,
    APIResponseValidationError,
)


# =============================================================================
# SHIELDED RESOURCE WRAPPERS
# =============================================================================


class _ShieldedResource:
    """Base class for shielded resources with passthrough support."""

    _resource: Any

    def __getattr__(self, name: str) -> Any:
        """Passthrough any attribute not explicitly defined."""
        return getattr(self._resource, name)


class _ShieldedChatCompletions(_ShieldedResource):
    """Shielded chat.completions with passthrough for non-API methods."""

    def __init__(self, completions: Any) -> None:
        self._resource = completions

    @raises(*_API_EXCEPTIONS)
    def create(self, **kwargs: Any) -> Any:
        """Create a chat completion with exhaustive error handling."""
        return self._resource.create(**kwargs)


class _ShieldedChat(_ShieldedResource):
    """Shielded chat resource."""

    def __init__(self, chat: Any) -> None:
        self._resource = chat
        self.completions = _ShieldedChatCompletions(chat.completions)


class _ShieldedEmbeddings(_ShieldedResource):
    """Shielded embeddings resource."""

    def __init__(self, embeddings: Any) -> None:
        self._resource = embeddings

    @raises(*_API_EXCEPTIONS)
    def create(self, **kwargs: Any) -> Any:
        """Create embeddings with exhaustive error handling."""
        return self._resource.create(**kwargs)


class _ShieldedImages(_ShieldedResource):
    """Shielded images resource."""

    def __init__(self, images: Any) -> None:
        self._resource = images

    @raises(*_API_EXCEPTIONS, ContentFilterFinishReasonError)
    def generate(self, **kwargs: Any) -> Any:
        """Generate images with exhaustive error handling."""
        return self._resource.generate(**kwargs)

    @raises(*_API_EXCEPTIONS)
    def edit(self, **kwargs: Any) -> Any:
        """Edit images with exhaustive error handling."""
        return self._resource.edit(**kwargs)

    @raises(*_API_EXCEPTIONS)
    def create_variation(self, **kwargs: Any) -> Any:
        """Create image variations with exhaustive error handling."""
        return self._resource.create_variation(**kwargs)


class _ShieldedTranscriptions(_ShieldedResource):
    """Shielded audio.transcriptions resource."""

    def __init__(self, transcriptions: Any) -> None:
        self._resource = transcriptions

    @raises(*_API_EXCEPTIONS)
    def create(self, **kwargs: Any) -> Any:
        """Transcribe audio with exhaustive error handling."""
        return self._resource.create(**kwargs)


class _ShieldedTranslations(_ShieldedResource):
    """Shielded audio.translations resource."""

    def __init__(self, translations: Any) -> None:
        self._resource = translations

    @raises(*_API_EXCEPTIONS)
    def create(self, **kwargs: Any) -> Any:
        """Translate audio with exhaustive error handling."""
        return self._resource.create(**kwargs)


class _ShieldedSpeech(_ShieldedResource):
    """Shielded audio.speech resource."""

    def __init__(self, speech: Any) -> None:
        self._resource = speech

    @raises(*_API_EXCEPTIONS)
    def create(self, **kwargs: Any) -> Any:
        """Generate speech with exhaustive error handling."""
        return self._resource.create(**kwargs)


class _ShieldedAudio(_ShieldedResource):
    """Shielded audio resource."""

    def __init__(self, audio: Any) -> None:
        self._resource = audio
        self.transcriptions = _ShieldedTranscriptions(audio.transcriptions)
        self.translations = _ShieldedTranslations(audio.translations)
        self.speech = _ShieldedSpeech(audio.speech)


class _ShieldedModerations(_ShieldedResource):
    """Shielded moderations resource."""

    def __init__(self, moderations: Any) -> None:
        self._resource = moderations

    @raises(*_API_EXCEPTIONS)
    def create(self, **kwargs: Any) -> Any:
        """Check content moderation with exhaustive error handling."""
        return self._resource.create(**kwargs)


class _ShieldedFiles(_ShieldedResource):
    """Shielded files resource."""

    def __init__(self, files: Any) -> None:
        self._resource = files

    @raises(*_API_EXCEPTIONS)
    def create(self, **kwargs: Any) -> Any:
        """Upload a file with exhaustive error handling."""
        return self._resource.create(**kwargs)

    @raises(*_API_EXCEPTIONS)
    def retrieve(self, file_id: str, **kwargs: Any) -> Any:
        """Retrieve file metadata with exhaustive error handling."""
        return self._resource.retrieve(file_id, **kwargs)

    @raises(*_API_EXCEPTIONS)
    def delete(self, file_id: str, **kwargs: Any) -> Any:
        """Delete a file with exhaustive error handling."""
        return self._resource.delete(file_id, **kwargs)

    @raises(*_API_EXCEPTIONS)
    def list(self, **kwargs: Any) -> Any:
        """List files with exhaustive error handling."""
        return self._resource.list(**kwargs)

    @raises(*_API_EXCEPTIONS)
    def content(self, file_id: str, **kwargs: Any) -> Any:
        """Get file content with exhaustive error handling."""
        return self._resource.content(file_id, **kwargs)


class _ShieldedModels(_ShieldedResource):
    """Shielded models resource."""

    def __init__(self, models: Any) -> None:
        self._resource = models

    @raises(*_API_EXCEPTIONS)
    def list(self, **kwargs: Any) -> Any:
        """List models with exhaustive error handling."""
        return self._resource.list(**kwargs)

    @raises(*_API_EXCEPTIONS)
    def retrieve(self, model: str, **kwargs: Any) -> Any:
        """Retrieve model info with exhaustive error handling."""
        return self._resource.retrieve(model, **kwargs)

    @raises(*_API_EXCEPTIONS)
    def delete(self, model: str, **kwargs: Any) -> Any:
        """Delete a fine-tuned model with exhaustive error handling."""
        return self._resource.delete(model, **kwargs)


class _ShieldedCompletions(_ShieldedResource):
    """Shielded completions resource (legacy)."""

    def __init__(self, completions: Any) -> None:
        self._resource = completions

    @raises(*_API_EXCEPTIONS)
    def create(self, **kwargs: Any) -> Any:
        """Create completion with exhaustive error handling."""
        return self._resource.create(**kwargs)


class _ShieldedFineTuningJobs(_ShieldedResource):
    """Shielded fine_tuning.jobs resource."""

    def __init__(self, jobs: Any) -> None:
        self._resource = jobs

    @raises(*_API_EXCEPTIONS)
    def create(self, **kwargs: Any) -> Any:
        """Create fine-tuning job with exhaustive error handling."""
        return self._resource.create(**kwargs)

    @raises(*_API_EXCEPTIONS)
    def retrieve(self, fine_tuning_job_id: str, **kwargs: Any) -> Any:
        """Retrieve fine-tuning job with exhaustive error handling."""
        return self._resource.retrieve(fine_tuning_job_id, **kwargs)

    @raises(*_API_EXCEPTIONS)
    def list(self, **kwargs: Any) -> Any:
        """List fine-tuning jobs with exhaustive error handling."""
        return self._resource.list(**kwargs)

    @raises(*_API_EXCEPTIONS)
    def cancel(self, fine_tuning_job_id: str, **kwargs: Any) -> Any:
        """Cancel fine-tuning job with exhaustive error handling."""
        return self._resource.cancel(fine_tuning_job_id, **kwargs)

    @raises(*_API_EXCEPTIONS)
    def list_events(self, fine_tuning_job_id: str, **kwargs: Any) -> Any:
        """List fine-tuning events with exhaustive error handling."""
        return self._resource.list_events(fine_tuning_job_id, **kwargs)


class _ShieldedFineTuning(_ShieldedResource):
    """Shielded fine_tuning resource."""

    def __init__(self, fine_tuning: Any) -> None:
        self._resource = fine_tuning
        self.jobs = _ShieldedFineTuningJobs(fine_tuning.jobs)


class _ShieldedBatches(_ShieldedResource):
    """Shielded batches resource."""

    def __init__(self, batches: Any) -> None:
        self._resource = batches

    @raises(*_API_EXCEPTIONS)
    def create(self, **kwargs: Any) -> Any:
        """Create batch with exhaustive error handling."""
        return self._resource.create(**kwargs)

    @raises(*_API_EXCEPTIONS)
    def retrieve(self, batch_id: str, **kwargs: Any) -> Any:
        """Retrieve batch with exhaustive error handling."""
        return self._resource.retrieve(batch_id, **kwargs)

    @raises(*_API_EXCEPTIONS)
    def list(self, **kwargs: Any) -> Any:
        """List batches with exhaustive error handling."""
        return self._resource.list(**kwargs)

    @raises(*_API_EXCEPTIONS)
    def cancel(self, batch_id: str, **kwargs: Any) -> Any:
        """Cancel batch with exhaustive error handling."""
        return self._resource.cancel(batch_id, **kwargs)


# =============================================================================
# SHIELDED CLIENTS
# =============================================================================


class OpenAI:
    """Shielded OpenAI client - drop-in replacement for openai.OpenAI.

    API methods that make network calls are decorated with @raises for
    exhaustive error handling. All other attributes pass through to the
    underlying client.
    """

    def __init__(self, **kwargs: Any) -> None:
        """Create a shielded OpenAI client.

        Accepts all arguments that openai.OpenAI accepts.
        """
        self._client = _openai.OpenAI(**kwargs)

        # Shielded resources (API calls with @raises)
        self.chat = _ShieldedChat(self._client.chat)
        self.embeddings = _ShieldedEmbeddings(self._client.embeddings)
        self.images = _ShieldedImages(self._client.images)
        self.audio = _ShieldedAudio(self._client.audio)
        self.moderations = _ShieldedModerations(self._client.moderations)
        self.files = _ShieldedFiles(self._client.files)
        self.models = _ShieldedModels(self._client.models)
        self.completions = _ShieldedCompletions(self._client.completions)
        self.fine_tuning = _ShieldedFineTuning(self._client.fine_tuning)
        self.batches = _ShieldedBatches(self._client.batches)

    def __getattr__(self, name: str) -> Any:
        """Passthrough any attribute not explicitly defined."""
        return getattr(self._client, name)

    def copy(self, **kwargs: Any) -> "OpenAI":
        """Create a copy with modified settings."""
        new_client = OpenAI.__new__(OpenAI)
        new_client._client = self._client.copy(**kwargs)
        new_client.chat = _ShieldedChat(new_client._client.chat)
        new_client.embeddings = _ShieldedEmbeddings(new_client._client.embeddings)
        new_client.images = _ShieldedImages(new_client._client.images)
        new_client.audio = _ShieldedAudio(new_client._client.audio)
        new_client.moderations = _ShieldedModerations(new_client._client.moderations)
        new_client.files = _ShieldedFiles(new_client._client.files)
        new_client.models = _ShieldedModels(new_client._client.models)
        new_client.completions = _ShieldedCompletions(new_client._client.completions)
        new_client.fine_tuning = _ShieldedFineTuning(new_client._client.fine_tuning)
        new_client.batches = _ShieldedBatches(new_client._client.batches)
        return new_client

    with_options = copy

    def close(self) -> None:
        """Close the client."""
        self._client.close()

    def __enter__(self) -> "OpenAI":
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()


class AsyncOpenAI:
    """Shielded AsyncOpenAI client - drop-in replacement for openai.AsyncOpenAI.

    API methods that make network calls are decorated with @raises for
    exhaustive error handling. All other attributes pass through to the
    underlying client.
    """

    def __init__(self, **kwargs: Any) -> None:
        """Create a shielded async OpenAI client.

        Accepts all arguments that openai.AsyncOpenAI accepts.
        """
        self._client = _openai.AsyncOpenAI(**kwargs)

        # Shielded resources (API calls with @raises)
        self.chat = _ShieldedChat(self._client.chat)
        self.embeddings = _ShieldedEmbeddings(self._client.embeddings)
        self.images = _ShieldedImages(self._client.images)
        self.audio = _ShieldedAudio(self._client.audio)
        self.moderations = _ShieldedModerations(self._client.moderations)
        self.files = _ShieldedFiles(self._client.files)
        self.models = _ShieldedModels(self._client.models)
        self.completions = _ShieldedCompletions(self._client.completions)
        self.fine_tuning = _ShieldedFineTuning(self._client.fine_tuning)
        self.batches = _ShieldedBatches(self._client.batches)

    def __getattr__(self, name: str) -> Any:
        """Passthrough any attribute not explicitly defined."""
        return getattr(self._client, name)

    def copy(self, **kwargs: Any) -> "AsyncOpenAI":
        """Create a copy with modified settings."""
        new_client = AsyncOpenAI.__new__(AsyncOpenAI)
        new_client._client = self._client.copy(**kwargs)
        new_client.chat = _ShieldedChat(new_client._client.chat)
        new_client.embeddings = _ShieldedEmbeddings(new_client._client.embeddings)
        new_client.images = _ShieldedImages(new_client._client.images)
        new_client.audio = _ShieldedAudio(new_client._client.audio)
        new_client.moderations = _ShieldedModerations(new_client._client.moderations)
        new_client.files = _ShieldedFiles(new_client._client.files)
        new_client.models = _ShieldedModels(new_client._client.models)
        new_client.completions = _ShieldedCompletions(new_client._client.completions)
        new_client.fine_tuning = _ShieldedFineTuning(new_client._client.fine_tuning)
        new_client.batches = _ShieldedBatches(new_client._client.batches)
        return new_client

    with_options = copy

    async def close(self) -> None:
        """Close the client."""
        await self._client.close()

    async def __aenter__(self) -> "AsyncOpenAI":
        return self

    async def __aexit__(self, *args: Any) -> None:
        await self.close()


# Aliases for compatibility
Client = OpenAI
AsyncClient = AsyncOpenAI


# =============================================================================
# MODULE-LEVEL PASSTHROUGH
# =============================================================================


def __getattr__(name: str) -> Any:
    """Passthrough any module attribute not explicitly defined.

    This allows `from pyrethrin.shields.openai import SomeType` to work
    for any type exported by the openai package.
    """
    return getattr(_openai, name)


# =============================================================================
# __all__ - Public API
# =============================================================================

__all__ = [
    # Version info
    "__shield_version__",
    "__openai_version__",
    # Shielded clients
    "OpenAI",
    "AsyncOpenAI",
    "Client",
    "AsyncClient",
    # Exceptions (re-exported for convenience)
    "OpenAIError",
    "APIError",
    "APIStatusError",
    "APIConnectionError",
    "APITimeoutError",
    "APIResponseValidationError",
    "BadRequestError",
    "AuthenticationError",
    "PermissionDeniedError",
    "NotFoundError",
    "ConflictError",
    "UnprocessableEntityError",
    "RateLimitError",
    "InternalServerError",
    "LengthFinishReasonError",
    "ContentFilterFinishReasonError",
    "InvalidWebhookSignatureError",
    # Common exports
    "NOT_GIVEN",
    "NotGiven",
    "Timeout",
    "RequestOptions",
    "Stream",
    "AsyncStream",
    "BaseModel",
    "file_from_path",
    "types",
]
