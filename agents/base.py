"""
Base Agent Class

Provides common functionality for all specialized agents.
"""

from abc import ABC, abstractmethod
from typing import List
from pathlib import Path
from langchain_openai import ChatOpenAI
from langchain_core.tools import BaseTool
from langchain_core.messages import SystemMessage


class BaseAgent(ABC):
    """
    Abstract base class for specialized agents.
    
    Provides common initialization and tool binding functionality.
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
        tools: List[BaseTool],
        model_name: str = "gpt-4o",
        temperature: float = 0.1
    ):
        """
        Initialize the agent with tools and model configuration.
        
        Args:
            tools: List of tools available to this agent
            model_name: OpenAI model to use
            temperature: Model temperature for response generation
        """
        self.tools = tools
        self.model = ChatOpenAI(model=model_name, temperature=temperature)
        self.model_with_tools = self.model.bind_tools(tools) if tools else self.model
    
    @property
    @abstractmethod
    def system_prompt(self) -> str:
        """Return the system prompt for this agent."""
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Return the agent name for logging."""
        pass
    
    def get_system_message(self) -> SystemMessage:
        """Return the system message for this agent."""
        return SystemMessage(content=self.system_prompt)
