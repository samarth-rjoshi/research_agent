"""
Multi-Agent Module

Contains specialized agents for the research pipeline:
- ResearcherAgent: Gathers information from web sources
- WriterAgent: Synthesizes research into documents
- ReviewerAgent: Fact-checks and edits documents
"""

from .researcher import ResearcherAgent
from .writer import WriterAgent
from .reviewer import ReviewerAgent
from .state import AgentState

__all__ = ["ResearcherAgent", "WriterAgent", "ReviewerAgent", "AgentState"]
