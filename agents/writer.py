"""
Writer Agent

Specializes in synthesizing research into well-structured documents.
Handles both initial drafts and revisions based on human feedback.
"""

from langchain.agents import create_agent
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
        print("\n✍️ WRITER AGENT Starting...")

        research_data = state.get("research_data", "")
        human_feedback = state.get("human_feedback", "")
        rewrite_instructions = state.get("rewrite_instructions", "")
        existing_draft = state.get("draft_document", "")
        original_query = state["messages"][0].content if state["messages"] else ""

        # Determine mode
        if rewrite_instructions and existing_draft:
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

Revise the document and save it using the write_document tool.
"""
        else:
            writing_prompt = f"""Based on the following research data, create a comprehensive document.

ORIGINAL REQUEST:
{original_query}

RESEARCH DATA:
{research_data}

Synthesize this into a well-structured markdown document and save it using the write_document tool.
"""

        # ✅ Create tool-enabled agent
        agent = create_agent(
            model=self.model,
            tools=self.tools,
            system_prompt=self.system_prompt,
        )

        # ✅ Let LangChain handle tool calls
        result = await agent.ainvoke({
            "messages": [HumanMessage(content=writing_prompt)]
        })

        # Final message
        final_message = result["messages"][-1]
        draft = final_message.content

        mode = "Revised" if rewrite_instructions else "Created initial"
        print(f"✅ WRITER AGENT Complete - {mode} draft document")

        return {
            "messages": result["messages"],
            "draft_document": draft,
            "current_phase": "human_review",
            "rewrite_instructions": "",  # Clear after applying
        }
