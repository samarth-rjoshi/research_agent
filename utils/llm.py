"""
LLM Factory

Single place to create the ChatOpenAI instance used by all agents.
"""

import os
from langchain_openai import ChatOpenAI


def get_llm(temperature: float = 0.1) -> ChatOpenAI:
    """
    Create a ChatOpenAI instance from environment variables.

    Args:
        temperature: Model temperature (default 0.1)

    Returns:
        Configured ChatOpenAI instance
    """
    return ChatOpenAI(
        model=os.environ.get("MODEL_NAME"),
        base_url=os.environ.get("OPENAI_API_BASE"),
        temperature=temperature,
    )
