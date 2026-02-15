"""
Researcher Agent

Specializes in gathering information from web sources, Wikipedia, and URLs.
Uses LangChain's create_agent for the tool-calling loop.
"""

from typing import List
from langchain_core.tools import BaseTool
from langchain_core.messages import HumanMessage

from langchain.agents import create_agent

from .base import BaseAgent
from .state import AgentState
from prompts import load_prompt


class ResearcherAgent(BaseAgent):
    """
    Researcher Agent - Gathers comprehensive information on a topic.

    Uses web search, Wikipedia, and URL fetching tools to collect
    raw research data from multiple sources.
    """

    def __init__(
        self,
        tools: List[BaseTool],
        temperature: float = 0.1,
    ):
        super().__init__(tools=tools, temperature=temperature)

    @property
    def name(self) -> str:
        return "Researcher"

    @property
    def system_prompt(self) -> str:
        return self.load_prompt("researcher_system")

    def _create_agent(self):
        """Create a LangChain agent with tools and system prompt."""
        return create_agent(
            self.model,
            tools=self.tools,
            system_prompt=load_prompt("researcher"),
        )

    async def run(self, state: AgentState) -> dict:
        """
        Execute the researcher agent.

        Args:
            state: Current agent state

        Returns:
            Updated state with research_data
        """
        print("\n🔍 RESEARCHER AGENT Starting...")

        agent = self._create_agent()

        result = await agent.ainvoke(
            {"messages": list(state["messages"])},
        )

        # Extract research data from the agent's final AI message
        final_messages = result["messages"]
        research_data = ""
        for msg in reversed(final_messages):
            if hasattr(msg, "content") and msg.content and not getattr(msg, "tool_calls", None):
                research_data = msg.content
                break

        print(f"✅ RESEARCHER AGENT Complete - Gathered {len(research_data)} chars of research")

        return {
            "messages": final_messages,
            "parallel_results": [research_data],
        }
