"""
Agent State Definition

Defines the shared state passed between agents in the pipeline.
"""

import operator
from typing import TypedDict, Annotated, List, Optional
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    """
    Shared state for the multi-agent pipeline.

    Attributes:
        messages: Annotated[List[BaseMessage], add_messages]
        research_data: str - Final aggregated research data
        parallel_results: Annotated[List[str], operator.add] - Individual research snippets from fan-out
        draft_document: str
        subtopics: List[str]
        human_feedback: str
        current_phase: str
    """
    messages: Annotated[List[BaseMessage], add_messages]
    research_data: str
    parallel_results: Annotated[List[str], operator.add]
    draft_document: str
    subtopics: List[str]
    human_feedback: str
    rewrite_instructions: str
    current_phase: str
