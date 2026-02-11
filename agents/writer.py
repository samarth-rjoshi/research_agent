"""
Writer Agent

Specializes in synthesizing research into well-structured documents.
Handles both initial drafts and revisions based on human feedback.
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
    On revision loops, incorporates human feedback to improve the draft.
    """

    @property
    def name(self) -> str:
        return "Writer"

    @property
    def system_prompt(self) -> str:
        return self.load_prompt("writer_system")

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

        mode = "Revised" if rewrite_instructions else "Created initial"
        print(f"✅ WRITER AGENT Complete - {mode} draft document")

        return {
            "messages": writer_messages,
            "draft_document": draft,
            "current_phase": "human_review",
            "rewrite_instructions": "",  # Clear instructions after incorporating them
        }
