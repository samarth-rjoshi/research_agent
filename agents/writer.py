"""
Writer Agent

Synthesizes research into well-structured documents.
Handles both initial drafts and revisions based on human feedback.
Non-agentic: performs a single LLM call and saves the output via a local utility.
"""

from langchain_core.messages import HumanMessage, SystemMessage

from .state import AgentState
from prompts import load_prompt
from utils import get_llm


async def run_writer(state: AgentState, tools: list = None) -> dict:
    """
    Execute the writer by generating content.

    Args:
        state: Current agent state with research_data and optional human_feedback
        tools: Ignored (kept for backward compatibility with the signature)

    Returns:
        Updated state with draft_document
    """
    print("\n✍️ WRITER Starting...")

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

                Please revise the document and provide the full updated content directly."""
    else:
        # Initial draft mode
        writing_prompt = f"""Based on the following research data, create a comprehensive document.

                ORIGINAL REQUEST:
                {original_query}

                RESEARCH DATA:
                {research_data}

                Please synthesize this into a well-structured document and provide the full content directly."""

    model = get_llm(temperature=0)
    system_prompt = load_prompt("writer")

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=writing_prompt),
    ]

    # Simple single LLM call
    response = await model.ainvoke(messages)
    draft = response.content

    messages.append(response)

    mode = "Revised" if rewrite_instructions else "Created initial"
    print(f"✅ WRITER Complete - {mode} draft document")

    return {
        "messages": messages,
        "draft_document": draft,
        "current_phase": "human_review",
        "rewrite_instructions": "",  # Clear instructions after incorporating them
    }
