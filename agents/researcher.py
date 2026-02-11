"""
Researcher Agent

Specializes in gathering information from web sources, Wikipedia, and URLs.
"""

from typing import List
from langchain_core.tools import BaseTool
from langchain_core.messages import HumanMessage, AIMessage

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
        
        messages = [self.get_system_message()] + list(state["messages"])
        
        # Run the agent loop
        research_messages = []
        current_messages = messages.copy()
        
        while True:
            response = self.model_with_tools.invoke(current_messages)
            research_messages.append(response)
            current_messages.append(response)
            
            # Check if agent wants to use tools
            if response.tool_calls:
                from langgraph.prebuilt import ToolNode
                tool_node = ToolNode(self.tools)
                tool_results = await tool_node.ainvoke({"messages": current_messages})
                
                for msg in tool_results["messages"]:
                    research_messages.append(msg)
                    current_messages.append(msg)
            else:
                # Agent is done researching
                break
        
        # Extract research data from final response
        research_data = response.content if hasattr(response, 'content') else str(response)
        
        print(f"✅ RESEARCHER AGENT Complete - Gathered {len(research_data)} chars of research")
        
        return {
            "messages": research_messages,
            "parallel_results": [research_data],
            "current_phase": "writing"
        }
