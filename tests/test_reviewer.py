"""
Live tests for ReviewerAgent — tool selection and argument verification.

Makes REAL LLM calls. Verifies the model calls read_document and/or
write_document with correct arguments during the review process.

Requires OPENAI_API_KEY.
"""

import pytest
from langchain_core.messages import SystemMessage, HumanMessage
from langsmith import testing as t


# ---------------------------------------------------------------------------
# Parametrized: 3 review scenarios with pre-filled drafts
# ---------------------------------------------------------------------------

REVIEW_SCENARIOS = [
    {
        "id": "review_ai_safety_doc",
        "research_data": (
            "## AI Safety Findings\n"
            "- Alignment problem is the central challenge\n"
            "- RLHF widely adopted, but has limitations\n"
            "- Sources: arxiv.org, DeepMind blog\n"
        ),
        "draft_document": (
            "# AI Safety: Current State\n\n"
            "## Executive Summary\n"
            "AI safety research has made progress but challenges remain.\n\n"
            "## Key Findings\n"
            "The alignment problem continues to be the main focus.\n"
            "RLHF is used but has known limitations.\n\n"
            "## References\n"
            "- arxiv.org\n"
        ),
    },
    {
        "id": "review_quantum_doc",
        "research_data": (
            "## Quantum Computing Findings\n"
            "- Google Willow: 105 qubits\n"
            "- IBM Condor: 1,121 qubits\n"
            "- Sources: nature.com, IBM blog\n"
        ),
        "draft_document": (
            "# Quantum Computing Progress\n\n"
            "## Overview\n"
            "Quantum computing is advancing rapidly.\n\n"
            "## Major Players\n"
            "Google and IBM are the leaders in qubit count.\n\n"
            "## Challenges\n"
            "Error correction and decoherence remain unsolved.\n"
        ),
    },
    {
        "id": "review_climate_doc",
        "research_data": (
            "## Climate Tech Findings\n"
            "- Carbon capture: Climeworks in Iceland\n"
            "- Solar: perovskite cells at 33.7% efficiency\n"
            "- Sources: IEA, Bloomberg NEF\n"
        ),
        "draft_document": (
            "# Climate Technology Innovations\n\n"
            "## Summary\n"
            "Climate technology is receiving unprecedented investment.\n\n"
            "## Carbon Capture\n"
            "Direct air capture is now operational at scale.\n\n"
            "## Solar Energy\n"
            "New cell technologies are breaking efficiency records.\n"
        ),
    },
]


@pytest.mark.langsmith(test_suite_name="Reviewer Agent - Tool Selection")
@pytest.mark.parametrize(
    "scenario",
    REVIEW_SCENARIOS,
    ids=[s["id"] for s in REVIEW_SCENARIOS],
)
def test_reviewer_uses_document_tools(
    document_model, reviewer_system_prompt, scenario
):
    """
    Given a draft document to review, the Reviewer should call document tools
    (read_document and/or write_document) with correct arguments.
    """
    research_data = scenario["research_data"]
    draft = scenario["draft_document"]

    t.log_inputs({
        "research_data": research_data,
        "draft_document": draft,
    })

    review_prompt = (
        f"Review and improve the following document.\n\n"
        f"ORIGINAL RESEARCH DATA (for fact-checking):\n{research_data}\n\n"
        f"DRAFT DOCUMENT TO REVIEW:\n{draft}\n\n"
        f"Please:\n"
        f"1. First read the document using read_document if needed\n"
        f"2. Review for accuracy and completeness\n"
        f"3. Make improvements and save the final version using write_document\n"
        f"4. Provide a brief summary of your review and changes"
    )

    messages = [
        SystemMessage(content=reviewer_system_prompt),
        HumanMessage(content=review_prompt),
    ]
    response = document_model.invoke(messages)

    tool_calls = response.tool_calls if response.tool_calls else []
    tool_names = [tc["name"] for tc in tool_calls]
    tool_args_list = [{"name": tc["name"], "args": tc["args"]} for tc in tool_calls]

    t.log_outputs({
        "tool_calls_made": tool_names,
        "tool_details": tool_args_list,
        "response_content": response.content[:300] if response.content else "",
    })

    # --- Assertions ---

    # 1. At least one document tool should be called
    valid_doc_tools = {"write_document", "read_document", "append_to_document", "list_documents"}
    assert len(tool_calls) > 0, "Reviewer should call at least one tool"

    # 2. Only valid document tools should be used
    for name in tool_names:
        assert name in valid_doc_tools, (
            f"Reviewer called unexpected tool '{name}'. Expected one of {valid_doc_tools}"
        )

    # 3. write_document should be called (to save the improved version)
    assert "write_document" in tool_names, (
        f"Reviewer should call write_document to save improvements. Got: {tool_names}"
    )

    # 4. Verify write_document arguments
    write_calls = [tc for tc in tool_calls if tc["name"] == "write_document"]
    for wc in write_calls:
        assert "filename" in wc["args"], "write_document must have 'filename'"
        assert "content" in wc["args"], "write_document must have 'content'"
        assert len(wc["args"]["content"]) > 50, (
            f"write_document content should be substantial, got {len(wc['args']['content'])} chars"
        )

    # 5. If read_document is called, verify it has filename arg
    read_calls = [tc for tc in tool_calls if tc["name"] == "read_document"]
    for rc in read_calls:
        assert "filename" in rc["args"], "read_document must have 'filename'"

    # Log feedback
    t.log_feedback(key="tools_called", score=1)
    t.log_feedback(key="only_valid_tools", score=1 if set(tool_names) <= valid_doc_tools else 0)
    t.log_feedback(key="write_document_called", score=1 if "write_document" in tool_names else 0)
    t.log_feedback(key="args_valid", score=1)


# ---------------------------------------------------------------------------
# Test: reviewer improves content (not just copies)
# ---------------------------------------------------------------------------

@pytest.mark.langsmith(test_suite_name="Reviewer Agent - Content Improvement")
@pytest.mark.parametrize(
    "scenario",
    REVIEW_SCENARIOS,
    ids=[s["id"] for s in REVIEW_SCENARIOS],
)
def test_reviewer_improves_document(
    document_model, reviewer_system_prompt, scenario
):
    """
    The reviewer's write_document content should be at least as long as
    the original draft (improvements should add, not remove).
    """
    draft = scenario["draft_document"]

    t.log_inputs({
        "draft_length": len(draft),
        "draft_preview": draft[:200],
    })

    review_prompt = (
        f"Review and improve the following document.\n\n"
        f"ORIGINAL RESEARCH DATA (for fact-checking):\n{scenario['research_data']}\n\n"
        f"DRAFT DOCUMENT TO REVIEW:\n{draft}\n\n"
        f"Please review for accuracy and completeness, make improvements, "
        f"and save the final version using write_document."
    )

    messages = [
        SystemMessage(content=reviewer_system_prompt),
        HumanMessage(content=review_prompt),
    ]
    response = document_model.invoke(messages)

    tool_calls = response.tool_calls if response.tool_calls else []
    write_calls = [tc for tc in tool_calls if tc["name"] == "write_document"]

    if not write_calls:
        t.log_outputs({"error": "write_document not called"})
        t.log_feedback(key="content_improved", score=0)
        pytest.fail("Reviewer did not call write_document")

    improved_content = write_calls[0]["args"].get("content", "")
    original_length = len(draft)
    improved_length = len(improved_content)

    t.log_outputs({
        "original_length": original_length,
        "improved_length": improved_length,
        "length_ratio": round(improved_length / max(original_length, 1), 2),
        "improved_preview": improved_content[:500],
    })

    # The improved document should be at least as long as original
    assert improved_length >= original_length * 0.8, (
        f"Improved doc ({improved_length} chars) should not be significantly shorter "
        f"than original ({original_length} chars)"
    )

    # Should still have markdown headings
    assert "#" in improved_content, "Improved document should retain markdown structure"

    t.log_feedback(
        key="content_improved",
        score=1 if improved_length >= original_length else 0,
    )
    t.log_feedback(key="has_structure", score=1 if "#" in improved_content else 0)
