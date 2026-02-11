"""
Live pipeline integration test.

Makes REAL LLM calls. Simulates the full Researcher → Writer → Reviewer
pipeline by chaining single LLM calls and verifying the output at each stage.

Requires OPENAI_API_KEY.
"""

import pytest
from langchain_core.messages import SystemMessage, HumanMessage
from langsmith import testing as t
from tests.conftest import RESEARCH_TOOLS, DOCUMENT_TOOLS


@pytest.mark.langsmith(test_suite_name="Full Pipeline - Integration")
def test_full_pipeline_tool_chain(
    llm,
    researcher_system_prompt,
    writer_system_prompt,
    reviewer_system_prompt,
):
    """
    End-to-end test: Researcher picks tools → Writer generates doc → Reviewer improves.
    Verifies the correct tools are selected at each stage.
    """
    query = "Research and write a report on the impact of large language models on software engineering"

    t.log_inputs({"query": query})

    # ─── Phase 1: Researcher ───
    research_model = llm.bind_tools(RESEARCH_TOOLS)
    research_response = research_model.invoke([
        SystemMessage(content=researcher_system_prompt),
        HumanMessage(content=query),
    ])

    research_tool_calls = research_response.tool_calls or []
    research_tool_names = [tc["name"] for tc in research_tool_calls]

    valid_research_tools = {"web_search", "fetch_webpage", "wikipedia_search"}
    assert len(research_tool_calls) > 0, "Researcher should call tools"
    for name in research_tool_names:
        assert name in valid_research_tools, f"Researcher used invalid tool: {name}"

    # ─── Phase 2: Writer ───
    # Simulate research data (since we didn't actually execute the tools)
    simulated_research = (
        "## Research on LLMs in Software Engineering\n"
        "- GitHub Copilot: increased developer productivity by 55% (GitHub study)\n"
        "- Code review automation: LLMs can catch 30% more bugs\n"
        "- Challenges: hallucinated code, security vulnerabilities\n"
        "- Key players: OpenAI Codex, Google Gemini Code, Anthropic Claude\n"
        "- Sources: GitHub blog, ACM research papers\n"
    )

    writing_prompt = (
        f"Based on the following research data, create a comprehensive document.\n\n"
        f"ORIGINAL REQUEST:\n{query}\n\n"
        f"RESEARCH DATA:\n{simulated_research}\n\n"
        f"Please synthesize this into a well-structured markdown document and "
        f"save it using the write_document tool."
    )

    document_model = llm.bind_tools(DOCUMENT_TOOLS)
    writer_response = document_model.invoke([
        SystemMessage(content=writer_system_prompt),
        HumanMessage(content=writing_prompt),
    ])

    writer_tool_calls = writer_response.tool_calls or []
    writer_tool_names = [tc["name"] for tc in writer_tool_calls]

    assert "write_document" in writer_tool_names, "Writer must call write_document"

    write_call = [tc for tc in writer_tool_calls if tc["name"] == "write_document"][0]
    written_content = write_call["args"].get("content", "")
    written_filename = write_call["args"].get("filename", "")

    assert len(written_content) > 100, "Written document should be substantial"
    assert len(written_filename) > 0, "Filename should not be empty"

    # ─── Phase 3: Reviewer ───
    review_prompt = (
        f"Review and improve the following document.\n\n"
        f"ORIGINAL RESEARCH DATA (for fact-checking):\n{simulated_research}\n\n"
        f"DRAFT DOCUMENT TO REVIEW:\n{written_content[:3000]}\n\n"
        f"Please review for accuracy and completeness, make improvements, "
        f"and save the final version using write_document."
    )

    reviewer_response = document_model.invoke([
        SystemMessage(content=reviewer_system_prompt),
        HumanMessage(content=review_prompt),
    ])

    reviewer_tool_calls = reviewer_response.tool_calls or []
    reviewer_tool_names = [tc["name"] for tc in reviewer_tool_calls]

    assert "write_document" in reviewer_tool_names, "Reviewer must call write_document"

    reviewed_call = [tc for tc in reviewer_tool_calls if tc["name"] == "write_document"][0]
    reviewed_content = reviewed_call["args"].get("content", "")

    assert len(reviewed_content) > 100, "Reviewed document should be substantial"

    # ─── Log full pipeline results ───
    t.log_outputs({
        "phase_1_tools": research_tool_names,
        "phase_1_args": [tc["args"] for tc in research_tool_calls],
        "phase_2_tools": writer_tool_names,
        "phase_2_filename": written_filename,
        "phase_2_content_length": len(written_content),
        "phase_3_tools": reviewer_tool_names,
        "phase_3_content_length": len(reviewed_content),
    })

    t.log_feedback(key="researcher_used_tools", score=1)
    t.log_feedback(key="writer_wrote_document", score=1)
    t.log_feedback(key="reviewer_improved_document", score=1)
    t.log_feedback(key="pipeline_complete", score=1)
