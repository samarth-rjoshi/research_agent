"""
Researcher Agent

Specializes in gathering information from web sources, Wikipedia, and URLs.
"""

from typing import List
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.tools import BaseTool

from .base import BaseAgent
from .state import AgentState


class ResearcherAgent(BaseAgent):
    """
    Researcher Agent - Gathers comprehensive information on a topic.

    Uses web search, Wikipedia, and URL fetching tools to collect
    raw research data from multiple sources.
    """

    @property
    def name(self) -> str:
        return "Researcher"

    @property
    def system_prompt(self) -> str:
        return self.load_prompt("researcher_system")

    async def run(self, state: AgentState) -> dict:
        """
        Execute the researcher agent.

        Args:
            state: Current agent state

        Returns:
            Updated state with research_data
        """

        print("\n🔍 RESEARCHER AGENT Starting...")

        # Create agent using LangChain's built-in agent factory
        agent = create_agent(
            model=self.model,
            tools=self.tools,
            system_prompt=self.system_prompt,
        )

        # Invoke agent with existing messages
        result = await agent.ainvoke({
            "messages": list(state["messages"])
        })

        # Extract final content
        final_message = result["messages"][-1]
        research_data = final_message.content

        print(f"✅ RESEARCHER AGENT Complete - Gathered {len(research_data)} chars of research")

        return {
            "messages": result["messages"],
            "parallel_results": [research_data],
            "current_phase": "writing"
        }
