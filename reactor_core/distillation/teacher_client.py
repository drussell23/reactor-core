"""
Teacher model clients for knowledge distillation.

Provides:
- Async OpenAI client (GPT-4o, GPT-4-turbo)
- Async Anthropic client (Claude-3)
- Unified interface for teacher model interactions
- Automatic retry and error handling
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, AsyncIterator, Dict, List, Optional, Union

logger = logging.getLogger(__name__)


class ModelType(Enum):
    """Supported teacher models."""
    # OpenAI
    GPT_4O = "gpt-4o"
    GPT_4O_MINI = "gpt-4o-mini"
    GPT_4_TURBO = "gpt-4-turbo"
    GPT_4 = "gpt-4"
    GPT_35_TURBO = "gpt-3.5-turbo"

    # Anthropic
    CLAUDE_3_OPUS = "claude-3-opus-20240229"
    CLAUDE_3_SONNET = "claude-3-5-sonnet-20241022"
    CLAUDE_3_HAIKU = "claude-3-haiku-20240307"

    # Google Gemini
    GEMINI_15_PRO = "gemini-1.5-pro"
    GEMINI_15_FLASH = "gemini-1.5-flash"
    GEMINI_15_FLASH_8B = "gemini-1.5-flash-8b"
    GEMINI_20_FLASH = "gemini-2.0-flash-exp"


@dataclass
class TeacherResponse:
    """Response from teacher model."""
    content: str
    model: str
    input_tokens: int
    output_tokens: int
    total_tokens: int
    finish_reason: str
    latency_ms: float
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def success(self) -> bool:
        """Check if response was successful."""
        return self.finish_reason in ("stop", "end_turn", "complete")


class TeacherClient(ABC):
    """
    Abstract base class for teacher model clients.

    Provides unified interface for OpenAI and Anthropic.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-4o",
        max_retries: int = 3,
        timeout: float = 60.0,
        base_url: Optional[str] = None,
    ):
        """
        Initialize teacher client.

        Args:
            api_key: API key (or use env var)
            model: Model name/ID
            max_retries: Maximum retry attempts
            timeout: Request timeout in seconds
            base_url: Optional custom base URL
        """
        self.api_key = api_key
        self.model = model
        self.max_retries = max_retries
        self.timeout = timeout
        self.base_url = base_url

        self._client = None

    @abstractmethod
    async def complete(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 1024,
        **kwargs,
    ) -> TeacherResponse:
        """
        Generate completion from messages.

        Args:
            messages: List of message dicts with 'role' and 'content'
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            **kwargs: Additional model-specific parameters

        Returns:
            TeacherResponse with generated content
        """
        pass

    @abstractmethod
    async def complete_stream(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 1024,
        **kwargs,
    ) -> AsyncIterator[str]:
        """
        Stream completion from messages.

        Args:
            messages: List of message dicts
            temperature: Sampling temperature
            max_tokens: Maximum tokens
            **kwargs: Additional parameters

        Yields:
            Content chunks as they arrive
        """
        pass

    async def ask(
        self,
        prompt: str,
        system: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        **kwargs,
    ) -> str:
        """
        Simple question-answer interface.

        Args:
            prompt: User prompt
            system: Optional system message
            temperature: Sampling temperature
            max_tokens: Maximum tokens

        Returns:
            Generated response text
        """
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        response = await self.complete(
            messages,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs,
        )
        return response.content

    async def close(self) -> None:
        """Close the client and cleanup resources."""
        pass


class OpenAIClient(TeacherClient):
    """
    OpenAI API client for GPT models.

    Supports GPT-4o, GPT-4-turbo, and GPT-3.5-turbo.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-4o",
        max_retries: int = 3,
        timeout: float = 60.0,
        base_url: Optional[str] = None,
        organization: Optional[str] = None,
    ):
        """
        Initialize OpenAI client.

        Args:
            api_key: OpenAI API key (or use OPENAI_API_KEY env var)
            model: Model name (gpt-4o, gpt-4-turbo, etc.)
            max_retries: Maximum retry attempts
            timeout: Request timeout
            base_url: Optional custom base URL
            organization: Optional organization ID
        """
        super().__init__(api_key, model, max_retries, timeout, base_url)

        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.organization = organization or os.getenv("OPENAI_ORG_ID")

        if not self.api_key:
            raise ValueError(
                "OpenAI API key required. Set OPENAI_API_KEY or pass api_key"
            )

    def _get_client(self):
        """Get or create OpenAI client."""
        if self._client is None:
            try:
                from openai import AsyncOpenAI
            except ImportError:
                raise ImportError(
                    "OpenAI library required. Install with: pip install openai"
                )

            self._client = AsyncOpenAI(
                api_key=self.api_key,
                organization=self.organization,
                base_url=self.base_url,
                timeout=self.timeout,
                max_retries=self.max_retries,
            )
        return self._client

    async def complete(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 1024,
        **kwargs,
    ) -> TeacherResponse:
        """Generate completion using OpenAI API."""
        client = self._get_client()
        start_time = time.time()

        try:
            response = await client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs,
            )

            latency_ms = (time.time() - start_time) * 1000

            choice = response.choices[0]
            usage = response.usage

            return TeacherResponse(
                content=choice.message.content or "",
                model=response.model,
                input_tokens=usage.prompt_tokens,
                output_tokens=usage.completion_tokens,
                total_tokens=usage.total_tokens,
                finish_reason=choice.finish_reason,
                latency_ms=latency_ms,
                metadata={
                    "id": response.id,
                    "created": response.created,
                },
            )

        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            raise

    async def complete_stream(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 1024,
        **kwargs,
    ) -> AsyncIterator[str]:
        """Stream completion using OpenAI API."""
        client = self._get_client()

        try:
            stream = await client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=True,
                **kwargs,
            )

            async for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content

        except Exception as e:
            logger.error(f"OpenAI streaming error: {e}")
            raise

    async def close(self) -> None:
        """Close the OpenAI client."""
        if self._client is not None:
            await self._client.close()
            self._client = None


class AnthropicClient(TeacherClient):
    """
    Anthropic API client for Claude models.

    Supports Claude-3 Opus, Sonnet, and Haiku.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "claude-3-5-sonnet-20241022",
        max_retries: int = 3,
        timeout: float = 60.0,
        base_url: Optional[str] = None,
    ):
        """
        Initialize Anthropic client.

        Args:
            api_key: Anthropic API key (or use ANTHROPIC_API_KEY env var)
            model: Model name (claude-3-opus, claude-3-sonnet, etc.)
            max_retries: Maximum retry attempts
            timeout: Request timeout
            base_url: Optional custom base URL
        """
        super().__init__(api_key, model, max_retries, timeout, base_url)

        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")

        if not self.api_key:
            raise ValueError(
                "Anthropic API key required. Set ANTHROPIC_API_KEY or pass api_key"
            )

    def _get_client(self):
        """Get or create Anthropic client."""
        if self._client is None:
            try:
                from anthropic import AsyncAnthropic
            except ImportError:
                raise ImportError(
                    "Anthropic library required. Install with: pip install anthropic"
                )

            kwargs = {
                "api_key": self.api_key,
                "timeout": self.timeout,
                "max_retries": self.max_retries,
            }
            if self.base_url:
                kwargs["base_url"] = self.base_url

            self._client = AsyncAnthropic(**kwargs)
        return self._client

    async def complete(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 1024,
        **kwargs,
    ) -> TeacherResponse:
        """Generate completion using Anthropic API."""
        client = self._get_client()
        start_time = time.time()

        # Extract system message if present
        system = None
        filtered_messages = []
        for msg in messages:
            if msg["role"] == "system":
                system = msg["content"]
            else:
                filtered_messages.append(msg)

        try:
            api_kwargs = {
                "model": self.model,
                "messages": filtered_messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
            }
            if system:
                api_kwargs["system"] = system

            # Remove any OpenAI-specific kwargs
            kwargs.pop("frequency_penalty", None)
            kwargs.pop("presence_penalty", None)
            kwargs.pop("logit_bias", None)
            api_kwargs.update(kwargs)

            response = await client.messages.create(**api_kwargs)

            latency_ms = (time.time() - start_time) * 1000

            content = ""
            for block in response.content:
                if hasattr(block, "text"):
                    content += block.text

            return TeacherResponse(
                content=content,
                model=response.model,
                input_tokens=response.usage.input_tokens,
                output_tokens=response.usage.output_tokens,
                total_tokens=response.usage.input_tokens + response.usage.output_tokens,
                finish_reason=response.stop_reason or "stop",
                latency_ms=latency_ms,
                metadata={
                    "id": response.id,
                    "type": response.type,
                },
            )

        except Exception as e:
            logger.error(f"Anthropic API error: {e}")
            raise

    async def complete_stream(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 1024,
        **kwargs,
    ) -> AsyncIterator[str]:
        """Stream completion using Anthropic API."""
        client = self._get_client()

        # Extract system message
        system = None
        filtered_messages = []
        for msg in messages:
            if msg["role"] == "system":
                system = msg["content"]
            else:
                filtered_messages.append(msg)

        try:
            api_kwargs = {
                "model": self.model,
                "messages": filtered_messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
            }
            if system:
                api_kwargs["system"] = system

            kwargs.pop("frequency_penalty", None)
            kwargs.pop("presence_penalty", None)
            api_kwargs.update(kwargs)

            async with client.messages.stream(**api_kwargs) as stream:
                async for text in stream.text_stream:
                    yield text

        except Exception as e:
            logger.error(f"Anthropic streaming error: {e}")
            raise

    async def close(self) -> None:
        """Close the Anthropic client."""
        if self._client is not None:
            await self._client.close()
            self._client = None


class GeminiClient(TeacherClient):
    """
    Google Gemini API client.

    Supports Gemini 1.5 Pro, Gemini 1.5 Flash, and Gemini 2.0 Flash.
    Optimized for high-throughput knowledge synthesis.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gemini-1.5-flash",
        max_retries: int = 3,
        timeout: float = 60.0,
        base_url: Optional[str] = None,
    ):
        """
        Initialize Gemini client.

        Args:
            api_key: Google AI API key (or use GOOGLE_API_KEY env var)
            model: Model name (gemini-1.5-flash, gemini-1.5-pro, etc.)
            max_retries: Maximum retry attempts
            timeout: Request timeout
            base_url: Optional custom base URL (for Vertex AI)
        """
        super().__init__(api_key, model, max_retries, timeout, base_url)

        self.api_key = api_key or os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")

        if not self.api_key:
            raise ValueError(
                "Google API key required. Set GOOGLE_API_KEY or pass api_key"
            )

        self._model_instance = None

    def _get_client(self):
        """Get or create Gemini client."""
        if self._client is None:
            try:
                import google.generativeai as genai
            except ImportError:
                raise ImportError(
                    "Google Generative AI library required. "
                    "Install with: pip install google-generativeai"
                )

            genai.configure(api_key=self.api_key)
            self._client = genai
            self._model_instance = genai.GenerativeModel(self.model)

        return self._client

    async def complete(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 1024,
        **kwargs,
    ) -> TeacherResponse:
        """Generate completion using Gemini API."""
        self._get_client()
        start_time = time.time()

        # Convert messages to Gemini format
        gemini_messages = []
        system_instruction = None

        for msg in messages:
            role = msg["role"]
            content = msg["content"]

            if role == "system":
                system_instruction = content
            elif role == "user":
                gemini_messages.append({"role": "user", "parts": [content]})
            elif role == "assistant":
                gemini_messages.append({"role": "model", "parts": [content]})

        try:
            # Create model with system instruction if provided
            if system_instruction:
                model = self._client.GenerativeModel(
                    self.model,
                    system_instruction=system_instruction,
                )
            else:
                model = self._model_instance

            # Configure generation
            generation_config = {
                "temperature": temperature,
                "max_output_tokens": max_tokens,
                "top_p": kwargs.get("top_p", 0.95),
                "top_k": kwargs.get("top_k", 40),
            }

            # Handle safety settings
            safety_settings = kwargs.get("safety_settings", None)

            # Generate response
            response = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: model.generate_content(
                    gemini_messages,
                    generation_config=generation_config,
                    safety_settings=safety_settings,
                ),
            )

            latency_ms = (time.time() - start_time) * 1000

            # Extract content
            content = ""
            if response.candidates:
                for part in response.candidates[0].content.parts:
                    if hasattr(part, "text"):
                        content += part.text

            # Get token counts
            input_tokens = 0
            output_tokens = 0

            if hasattr(response, "usage_metadata"):
                input_tokens = getattr(response.usage_metadata, "prompt_token_count", 0)
                output_tokens = getattr(response.usage_metadata, "candidates_token_count", 0)

            # Determine finish reason
            finish_reason = "stop"
            if response.candidates:
                candidate = response.candidates[0]
                if hasattr(candidate, "finish_reason"):
                    fr = str(candidate.finish_reason)
                    if "STOP" in fr:
                        finish_reason = "stop"
                    elif "MAX_TOKENS" in fr:
                        finish_reason = "length"
                    elif "SAFETY" in fr:
                        finish_reason = "content_filter"
                    else:
                        finish_reason = fr.lower()

            return TeacherResponse(
                content=content,
                model=self.model,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                total_tokens=input_tokens + output_tokens,
                finish_reason=finish_reason,
                latency_ms=latency_ms,
                metadata={
                    "safety_ratings": (
                        [str(r) for r in response.candidates[0].safety_ratings]
                        if response.candidates and hasattr(response.candidates[0], "safety_ratings")
                        else []
                    ),
                },
            )

        except Exception as e:
            logger.error(f"Gemini API error: {e}")
            raise

    async def complete_stream(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 1024,
        **kwargs,
    ) -> AsyncIterator[str]:
        """Stream completion using Gemini API."""
        self._get_client()

        # Convert messages to Gemini format
        gemini_messages = []
        system_instruction = None

        for msg in messages:
            role = msg["role"]
            content = msg["content"]

            if role == "system":
                system_instruction = content
            elif role == "user":
                gemini_messages.append({"role": "user", "parts": [content]})
            elif role == "assistant":
                gemini_messages.append({"role": "model", "parts": [content]})

        try:
            # Create model with system instruction if provided
            if system_instruction:
                model = self._client.GenerativeModel(
                    self.model,
                    system_instruction=system_instruction,
                )
            else:
                model = self._model_instance

            generation_config = {
                "temperature": temperature,
                "max_output_tokens": max_tokens,
            }

            # Stream response
            response = model.generate_content(
                gemini_messages,
                generation_config=generation_config,
                stream=True,
            )

            for chunk in response:
                if chunk.text:
                    yield chunk.text

        except Exception as e:
            logger.error(f"Gemini streaming error: {e}")
            raise

    async def close(self) -> None:
        """Close the Gemini client."""
        self._client = None
        self._model_instance = None


def create_teacher_client(
    model: Union[str, ModelType],
    api_key: Optional[str] = None,
    **kwargs,
) -> TeacherClient:
    """
    Factory function to create appropriate teacher client.

    Args:
        model: Model name or ModelType enum
        api_key: API key (optional, uses env vars)
        **kwargs: Additional client arguments

    Returns:
        Appropriate TeacherClient instance
    """
    if isinstance(model, ModelType):
        model = model.value

    model_lower = model.lower()

    # Detect provider from model name
    if any(x in model_lower for x in ["gpt", "openai", "o1", "davinci", "turbo"]):
        return OpenAIClient(api_key=api_key, model=model, **kwargs)
    elif any(x in model_lower for x in ["claude", "anthropic"]):
        return AnthropicClient(api_key=api_key, model=model, **kwargs)
    elif any(x in model_lower for x in ["gemini", "google"]):
        return GeminiClient(api_key=api_key, model=model, **kwargs)
    else:
        # Default to OpenAI
        logger.warning(f"Unknown model {model}, defaulting to OpenAI client")
        return OpenAIClient(api_key=api_key, model=model, **kwargs)


# Convenience exports
__all__ = [
    "TeacherClient",
    "OpenAIClient",
    "AnthropicClient",
    "GeminiClient",
    "TeacherResponse",
    "ModelType",
    "create_teacher_client",
]
