"""
Agent State Definition

Defines the shared state passed between agents in the pipeline.
"""

from typing import TypedDict, Annotated, List
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    """
    Shared state for the multi-agent pipeline.
    
    Attributes:
        messages: Conversation history with message accumulation
        research_data: Raw research findings from Researcher Agent
        draft_document: Document draft from Writer Agent
        review_feedback: Feedback and edits from Reviewer Agent
        final_document: Final polished document
        current_phase: Current pipeline phase (research/writing/review)
    """
    messages: Annotated[List[BaseMessage], add_messages]
    research_data: str
    draft_document: str
    review_feedback: str
    final_document: str
    current_phase: str
