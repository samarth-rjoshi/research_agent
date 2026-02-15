"""
Writer Agent

Specializes in synthesizing research into well-structured documents.
Handles both initial drafts and revisions based on human feedback.
Uses LangChain's create_agent for the tool-calling loop.
"""

from typing import List
from langchain_core.tools import BaseTool
from langchain_core.messages import HumanMessage

from langchain.agents import create_agent

from .base import BaseAgent
from .state import AgentState
from prompts import load_prompt


class WriterAgent(BaseAgent):
    """
    Writer Agent - Synthesizes research into polished documents.

    Takes raw research data and creates well-structured, comprehensive
    documents with proper formatting and citations.
    On revision loops, incorporates human feedback to improve the draft.
    """

    def __init__(
        self,
        tools: List[BaseTool],
        temperature: float = 0.1,
    ):
        super().__init__(tools=tools, temperature=temperature)

    @property
    def name(self) -> str:
        return "Writer"

    @property
    def system_prompt(self) -> str:
        return self.load_prompt("writer_system")

    def _create_agent(self):
        """Create a LangChain agent with tools and system prompt."""
        return create_agent(
            self.model,
            tools=self.tools,
            system_prompt=load_prompt("writer"),
        )

    async def run(self, state: AgentState) -> dict:
        """
        Execute the writer agent.

        Args:
            state: Current agent state with research_data and optional human_feedback

        Returns:
            Updated state with draft_document
        """
        print("\n✍️ WRITER AGENT Starting...")

        research_data = state.get("research_data", "")
        human_feedback = state.get("human_feedback", "")
        rewrite_instructions = state.get("rewrite_instructions", "")
        existing_draft = state.get("draft_document", "")
        original_query = state["messages"][0].content if state["messages"] else ""

        if rewrite_instructions and existing_draft:
            # Revision mode — improve existing draft based on supervisor instructions
            writing_prompt = f"""Revise the following document based on the supervisor's instructions.

ORIGINAL REQUEST:
{original_query}

CURRENT DRAFT:
{existing_draft}

INSTRUCTIONS:
{rewrite_instructions}

(Reference) RAW HUMAN FEEDBACK:
{human_feedback}

(Reference) RESEARCH DATA:
{research_data[:3000]}

Please revise the document incorporating the instructions and save it using the write_document tool."""
        else:
            # Initial draft mode
            writing_prompt = f"""Based on the following research data, create a comprehensive document.

ORIGINAL REQUEST:
{original_query}

RESEARCH DATA:
{research_data}

Please synthesize this into a well-structured markdown document and save it using the write_document tool."""

        agent = self._create_agent()

        result = await agent.ainvoke(
            {"messages": [HumanMessage(content=writing_prompt)]},
            config={"recursion_limit": 10},  # ~5 tool calls max
        )

        # Extract draft from the agent's final AI message
        final_messages = result["messages"]
        draft = ""
        for msg in reversed(final_messages):
            if hasattr(msg, "content") and msg.content and not getattr(msg, "tool_calls", None):
                draft = msg.content
                break

        mode = "Revised" if rewrite_instructions else "Created initial"
        print(f"✅ WRITER AGENT Complete - {mode} draft document")

        return {
            "messages": final_messages,
            "draft_document": draft,
            "current_phase": "human_review",
            "rewrite_instructions": "",  # Clear instructions after incorporating them
        }
