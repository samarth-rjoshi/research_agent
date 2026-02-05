"""
Writer Agent

Specializes in synthesizing research into well-structured documents.
"""

from typing import List
from langchain_core.tools import BaseTool
from langchain_core.messages import HumanMessage

from .base import BaseAgent
from .state import AgentState


class WriterAgent(BaseAgent):
    """
    Writer Agent - Synthesizes research into polished documents.
    
    Takes raw research data and creates well-structured, comprehensive
    documents with proper formatting and citations.
    """
    
    @property
    def name(self) -> str:
        return "Writer"
    
    @property
    def system_prompt(self) -> str:
        return """You are a Document Writing Specialist Agent. Your job is to synthesize research into polished documents.

YOUR RESPONSIBILITIES:
1. Take the research data provided and synthesize it into a well-structured document
2. Use write_document to create the final document file
3. Ensure proper formatting, flow, and readability

DOCUMENT STRUCTURE:
- Title and date
- Executive Summary (2-3 paragraphs)
- Main sections with clear headings
- Key findings and insights
- Sources/References section

WRITING GUIDELINES:
- Write in clear, professional prose
- Use markdown formatting for structure
- Include all relevant facts from research
- Cite sources inline and in references
- Be comprehensive but concise

OUTPUT: Create a markdown document using the write_document tool."""

    async def run(self, state: AgentState) -> dict:
        """
        Execute the writer agent.
        
        Args:
            state: Current agent state with research_data
            
        Returns:
            Updated state with draft_document
        """
        print("\n✍️ WRITER AGENT Starting...")
        
        # Create a message with the research data
        research_data = state.get("research_data", "")
        original_query = state["messages"][0].content if state["messages"] else ""
        
        writing_prompt = f"""Based on the following research data, create a comprehensive document.

ORIGINAL REQUEST:
{original_query}

RESEARCH DATA:
{research_data}

Please synthesize this into a well-structured markdown document and save it using the write_document tool."""
        
        messages = [
            self.get_system_message(),
            HumanMessage(content=writing_prompt)
        ]
        
        # Run the agent loop
        writer_messages = []
        current_messages = messages.copy()
        
        while True:
            response = self.model_with_tools.invoke(current_messages)
            writer_messages.append(response)
            current_messages.append(response)
            
            # Check if agent wants to use tools
            if response.tool_calls:
                from langgraph.prebuilt import ToolNode
                tool_node = ToolNode(self.tools)
                tool_results = await tool_node.ainvoke({"messages": current_messages})
                
                for msg in tool_results["messages"]:
                    writer_messages.append(msg)
                    current_messages.append(msg)
            else:
                # Agent is done writing
                break
        
        draft = response.content if hasattr(response, 'content') else str(response)
        
        print(f"✅ WRITER AGENT Complete - Created draft document")
        
        return {
            "messages": writer_messages,
            "draft_document": draft,
            "current_phase": "review"
        }
