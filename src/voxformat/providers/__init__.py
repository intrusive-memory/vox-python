"""Provider adapters: attach TTS engine data from any source to a ``.vox``."""

from .base import (
    ProviderAdapter,
    ProviderEmbedding,
    ProviderSample,
    ProviderVoice,
    attach,
    detach,
    get_adapter,
    register_adapter,
    registered_adapters,
)
from .handles import AppleAdapter, ElevenLabsAdapter, OpenAIAdapter
from .gemini import GeminiAdapter, GeminiVoicesClient

__all__ = [
    "ProviderAdapter",
    "ProviderEmbedding",
    "ProviderSample",
    "ProviderVoice",
    "attach",
    "detach",
    "get_adapter",
    "register_adapter",
    "registered_adapters",
    "AppleAdapter",
    "ElevenLabsAdapter",
    "OpenAIAdapter",
    "GeminiAdapter",
    "GeminiVoicesClient",
]
