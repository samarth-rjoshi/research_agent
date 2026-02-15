"""
Base Agent Class

Provides common functionality for all specialized agents.
"""

import os
from pathlib import Path
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage


class BaseAgent:
    """
    Base class for specialized agents.

    Provides common initialization and prompt loading functionality.
    Model name and base URL are read from environment variables
    (MODEL_NAME, OPENAI_API_BASE) so all agents share a single config.
    """

    @staticmethod
    def load_prompt(prompt_name: str) -> str:
        """
        Load a prompt from the prompts/ directory.

        Args:
            prompt_name: Name of the prompt file (without .md extension)

        Returns:
            The content of the prompt file
        """
        project_root = Path(__file__).parent.parent
        prompt_path = project_root / "prompts" / f"{prompt_name}.md"

        try:
            with open(prompt_path, "r", encoding="utf-8") as f:
                return f.read().strip()
        except Exception as e:
            print(f"❌ Error loading prompt {prompt_name}: {str(e)}")
            return f"Prompt {prompt_name} not found."

    def __init__(
        self,
        tools=None,
        temperature: float = 0.1,
    ):
        """
        Initialize the agent with model configuration from environment.

        Args:
            tools: Optional list of tools available to this agent
            temperature: Model temperature for response generation
        """
        self.tools = tools or []
        self.model = ChatOpenAI(
            model=os.environ.get("MODEL_NAME"),
            base_url=os.environ.get("OPENAI_API_BASE"),
            temperature=temperature,
        )

    def get_system_message(self, system_prompt: str) -> SystemMessage:
        """Return the system message for this agent."""
        return SystemMessage(content=system_prompt)
