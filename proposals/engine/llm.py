"""
LLM Configuration - Sets up Gemini as the LLM for LlamaIndex

This module initializes and configures the Gemini LLM for use with LlamaIndex.
It also sets up fallback models and caching mechanisms.
"""

import os
from typing import List, Tuple, Callable, Any, Optional
import logging

from llama_index.core import Settings
from llama_index.llms.gemini import Gemini
from llama_index.embeddings.gemini import GeminiEmbedding
from llama_index.core.llms.callbacks import llm_completion_callback
import google.generativeai as genai
from llama_index.llms.gemini import Gemini

from llama_index.core import Settings


# LlamaIndex settings


def configure_settings():
    """Configure LlamaIndex settings"""

    # Reset any existing settings

    Settings.llm = Gemini()
    Settings.embed_model = GeminiEmbedding(model_name="models/embedding-001")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)



class LLMConfig:
    """
    Manages LLM configuration and setup
    """

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the LLM configuration

        Args:
            api_key: Optional Google API key. If None, uses GOOGLE_API_KEY env var
        """
        # Get API key from environment if not provided
        self.api_key = api_key or os.environ.get("GOOGLE_API_KEY")
        if not self.api_key:
            logger.warning(
                "No Google API key found. Set GOOGLE_API_KEY environment variable."
            )

        # Configure Gemini
        self._configure_gemini()
        configure_settings()

    def _configure_gemini(self):
        """Configure Gemini API and set up models"""
        if self.api_key:
            genai.configure(api_key=self.api_key)
            logger.info("Gemini API configured successfully")

    def get_default_llm(self):
        """
        Returns the default Gemini LLM instance
        """
        return Gemini(
            temperature=0.2,  # Lower temperature for more factual responses
            additional_kwargs={"top_p": 0.95},
        )

    def setup_llama_index(self):
        """
        Configure LlamaIndex to use Gemini
        """
        # Set the default LLM for LlamaIndex
        Settings.llm = self.get_default_llm()
        logger.info("LlamaIndex configured to use Gemini Pro")

        # Return the configured Settings for further customization
        return Settings

    @classmethod
    def list_available_models(cls):
        """
        List all available Gemini models
        """
        models = []
        try:
            for m in genai.list_models():
                if "generateContent" in m.supported_generation_methods:
                    models.append(m.name)
            return models
        except Exception as e:
            logger.error(f"Failed to list models: {e}")
            return []


def get_llm_with_fallbacks():
    """
    Creates a router with primary and fallback LLMs

    Not implemented yet - placeholder for future development
    """
    # This would integrate with a Router pattern to support multi-LLM fallback
    # For now, just return the default LLM
    return LLMConfig().get_default_llm()


# Automatic setup when module is imported
default_config = LLMConfig()
