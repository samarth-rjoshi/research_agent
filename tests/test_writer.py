"""
Live tests for WriterAgent — tool selection and argument verification.

Makes REAL LLM calls. Verifies the model calls write_document with
proper filename and non-empty content arguments.

Requires OPENAI_API_KEY.
"""

import pytest
from langchain_core.messages import SystemMessage, HumanMessage
from langsmith import testing as t


# ---------------------------------------------------------------------------
# Parametrized: 3 writing scenarios with pre-filled research data
# ---------------------------------------------------------------------------

WRITING_SCENARIOS = [
    {
        "id": "ai_safety_report",
        "query": "Write a comprehensive report on AI safety",
        "research_data": (
            "## AI Safety Research Findings\n"
            "- Leading labs: Anthropic (Constitutional AI), OpenAI (alignment team), DeepMind (scalable oversight)\n"
            "- Key challenges: alignment problem, reward hacking, distributional shift\n"
            "- Recent breakthroughs: RLHF, debate-based alignment, interpretability tools\n"
            "- Expert predictions: 50% of researchers expect AGI by 2040\n"
            "- Sources: arxiv.org, nature.com, MIT Tech Review\n"
        ),
    },
    {
        "id": "quantum_computing_summary",
        "query": "Write a summary document about quantum computing progress",
        "research_data": (
            "## Quantum Computing Research\n"
            "- Google Willow chip: 105 qubits, achieved quantum error correction milestone\n"
            "- IBM Condor: 1,121 qubit processor demonstrated\n"
            "- Key applications: cryptography, drug discovery, optimization\n"
            "- Challenges: decoherence, error rates, scalability\n"
            "- Timeline: practical quantum advantage expected 2027-2030\n"
            "- Sources: nature.com, IBM Research blog, Google AI blog\n"
        ),
    },
    {
        "id": "climate_tech_overview",
        "query": "Create a document about climate technology innovations",
        "research_data": (
            "## Climate Tech Findings\n"
            "- Carbon capture: Climeworks operational in Iceland, 4000 tons/year\n"
            "- Solar efficiency: perovskite cells reaching 33.7% efficiency\n"
            "- Battery tech: solid-state batteries approaching commercialization\n"
            "- Green hydrogen: costs dropping, $2/kg target by 2030\n"
            "- Investment: $70B in climate tech VC funding in 2024\n"
            "- Sources: IEA, Bloomberg NEF, Nature Energy\n"
        ),
    },
]


@pytest.mark.langsmith(test_suite_name="Writer Agent - Tool Selection")
@pytest.mark.parametrize(
    "scenario",
    WRITING_SCENARIOS,
    ids=[s["id"] for s in WRITING_SCENARIOS],
)
def test_writer_calls_write_document(
    document_model, writer_system_prompt, scenario
):
    """
    Given research data, the Writer should call write_document with a valid
    filename and substantial content.
    """
    query = scenario["query"]
    research_data = scenario["research_data"]

    t.log_inputs({"query": query, "research_data": research_data})

    writing_prompt = (
        f"Based on the following research data, create a comprehensive document.\n\n"
        f"ORIGINAL REQUEST:\n{query}\n\n"
        f"RESEARCH DATA:\n{research_data}\n\n"
        f"Please synthesize this into a well-structured markdown document and "
        f"save it using the write_document tool."
    )

    messages = [
        SystemMessage(content=writer_system_prompt),
        HumanMessage(content=writing_prompt),
    ]
    response = document_model.invoke(messages)

    tool_calls = response.tool_calls if response.tool_calls else []
    tool_names = [tc["name"] for tc in tool_calls]
    tool_args = {tc["name"]: tc["args"] for tc in tool_calls}

    t.log_outputs({
        "tool_calls_made": tool_names,
        "write_doc_filename": tool_args.get("write_document", {}).get("filename", "N/A"),
        "write_doc_content_length": len(
            tool_args.get("write_document", {}).get("content", "")
        ),
        "response_content": response.content[:200] if response.content else "",
    })

    # --- Assertions ---

    # 1. write_document must be called
    assert "write_document" in tool_names, (
        f"Writer must call write_document. Got: {tool_names}"
    )

    # 2. Find the write_document call
    write_calls = [tc for tc in tool_calls if tc["name"] == "write_document"]
    wd = write_calls[0]

    # 3. Must have filename argument
    assert "filename" in wd["args"], "write_document must have 'filename' argument"
    filename = wd["args"]["filename"]
    assert isinstance(filename, str) and len(filename) > 0, "filename must be non-empty string"

    # 4. Must have content argument
    assert "content" in wd["args"], "write_document must have 'content' argument"
    content = wd["args"]["content"]
    assert isinstance(content, str), "content must be a string"
    assert len(content) > 100, (
        f"Document content should be substantial (>100 chars), got {len(content)} chars"
    )

    # 5. No invalid tools should be called
    valid_doc_tools = {"write_document", "read_document", "append_to_document", "list_documents"}
    for name in tool_names:
        assert name in valid_doc_tools, (
            f"Writer called unexpected tool '{name}'. Expected one of {valid_doc_tools}"
        )

    # Log feedback
    t.log_feedback(key="write_document_called", score=1)
    t.log_feedback(key="filename_valid", score=1 if len(filename) > 0 else 0)
    t.log_feedback(key="content_substantial", score=1 if len(content) > 100 else 0)
    t.log_feedback(key="only_valid_tools", score=1 if set(tool_names) <= valid_doc_tools else 0)


# ---------------------------------------------------------------------------
# Test: document content quality
# ---------------------------------------------------------------------------

@pytest.mark.langsmith(test_suite_name="Writer Agent - Content Quality")
@pytest.mark.parametrize(
    "scenario",
    WRITING_SCENARIOS,
    ids=[s["id"] for s in WRITING_SCENARIOS],
)
def test_writer_document_has_structure(
    document_model, writer_system_prompt, scenario
):
    """
    The generated document content should have markdown structure
    (headings, sections) — not just a flat wall of text.
    """
    query = scenario["query"]
    research_data = scenario["research_data"]

    t.log_inputs({"query": query})

    writing_prompt = (
        f"Based on the following research data, create a comprehensive document.\n\n"
        f"ORIGINAL REQUEST:\n{query}\n\n"
        f"RESEARCH DATA:\n{research_data}\n\n"
        f"Please synthesize this into a well-structured markdown document and "
        f"save it using the write_document tool."
    )

    messages = [
        SystemMessage(content=writer_system_prompt),
        HumanMessage(content=writing_prompt),
    ]
    response = document_model.invoke(messages)

    tool_calls = response.tool_calls if response.tool_calls else []
    write_calls = [tc for tc in tool_calls if tc["name"] == "write_document"]

    if not write_calls:
        t.log_outputs({"error": "write_document not called"})
        t.log_feedback(key="has_headings", score=0)
        pytest.fail("write_document was not called")

    content = write_calls[0]["args"].get("content", "")

    # Check for markdown structure
    has_h1 = "# " in content
    has_h2 = "## " in content
    has_headings = has_h1 or has_h2
    line_count = len(content.strip().split("\n"))

    t.log_outputs({
        "content_length": len(content),
        "has_h1_heading": has_h1,
        "has_h2_heading": has_h2,
        "line_count": line_count,
        "content_preview": content[:500],
    })

    assert has_headings, "Document should contain markdown headings (# or ##)"
    assert line_count > 10, f"Document should have multiple lines, got {line_count}"

    t.log_feedback(key="has_headings", score=1 if has_headings else 0)
    t.log_feedback(key="sufficient_length", score=1 if line_count > 10 else 0)
