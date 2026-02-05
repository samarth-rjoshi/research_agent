"""
Reviewer Agent

Specializes in fact-checking and improving document quality.
"""

from typing import List
from langchain_core.tools import BaseTool
from langchain_core.messages import HumanMessage

from .base import BaseAgent
from .state import AgentState


class ReviewerAgent(BaseAgent):
    """
    Reviewer Agent - Fact-checks and improves documents.
    
    Reviews the draft document for accuracy, completeness, and quality.
    Makes improvements and produces the final polished version.
    """
    
    @property
    def name(self) -> str:
        return "Reviewer"
    
    @property
    def system_prompt(self) -> str:
        return """You are a Document Review Specialist Agent. Your job is to fact-check and improve documents.

YOUR RESPONSIBILITIES:
1. Read the current document using read_document
2. Review for accuracy, completeness, and quality
3. Make improvements and save the final version using write_document

REVIEW CHECKLIST:
- Verify facts are accurately represented
- Check that sources are properly cited
- Ensure logical flow and structure
- Improve clarity and readability
- Fix any grammatical or formatting issues
- Add any missing important information

OUTPUT:
1. Provide a brief review summary of changes made
2. Save the improved document using write_document

Be thorough but efficient. Focus on substantive improvements."""

    async def run(self, state: AgentState) -> dict:
        """
        Execute the reviewer agent.
        
        Args:
            state: Current agent state with draft_document
            
        Returns:
            Updated state with final_document
        """
        print("\n📝 REVIEWER AGENT Starting...")
        
        draft = state.get("draft_document", "")
        research_data = state.get("research_data", "")
        
        review_prompt = f"""Review and improve the following document.

ORIGINAL RESEARCH DATA (for fact-checking):
{research_data[:3000]}...

DRAFT DOCUMENT TO REVIEW:
{draft}

Please:
1. First read the document using read_document if needed
2. Review for accuracy and completeness
3. Make improvements and save the final version using write_document
4. Provide a brief summary of your review and changes"""
        
        messages = [
            self.get_system_message(),
            HumanMessage(content=review_prompt)
        ]
        
        # Run the agent loop
        review_messages = []
        current_messages = messages.copy()
        
        while True:
            response = self.model_with_tools.invoke(current_messages)
            review_messages.append(response)
            current_messages.append(response)
            
            # Check if agent wants to use tools
            if response.tool_calls:
                from langgraph.prebuilt import ToolNode
                tool_node = ToolNode(self.tools)
                tool_results = await tool_node.ainvoke({"messages": current_messages})
                
                for msg in tool_results["messages"]:
                    review_messages.append(msg)
                    current_messages.append(msg)
            else:
                # Agent is done reviewing
                break
        
        final = response.content if hasattr(response, 'content') else str(response)
        
        print(f"✅ REVIEWER AGENT Complete - Document finalized")
        
        return {
            "messages": review_messages,
            "review_feedback": final,
            "final_document": final,
            "current_phase": "complete"
        }
