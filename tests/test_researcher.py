"""
Live tests for ResearcherAgent — tool selection and argument verification.

Makes REAL LLM calls. Verifies the model picks the right research tools
(web_search, wikipedia_search, fetch_webpage) and generates correct arguments.

Requires OPENAI_API_KEY.
"""

import pytest
from langchain_core.messages import SystemMessage, HumanMessage
from langsmith import testing as t


# ---------------------------------------------------------------------------
# Parametrized: 4 diverse research queries
# ---------------------------------------------------------------------------

RESEARCH_QUERIES = [
    (
        "Research the current state of artificial general intelligence development",
        {"expected_tools": ["web_search", "wikipedia_search"]},
    ),
    (
        "Find recent statistics on global renewable energy adoption in 2025",
        {"expected_tools": ["web_search"]},
    ),
    (
        "What are the key features of the Rust programming language compared to C++?",
        {"expected_tools": ["web_search", "wikipedia_search"]},
    ),
    (
        "Get information about the James Webb Space Telescope latest discoveries",
        {"expected_tools": ["web_search", "wikipedia_search"]},
    ),
]


@pytest.mark.langsmith(test_suite_name="Researcher Agent - Tool Selection")
@pytest.mark.parametrize(
    "query, expectations",
    RESEARCH_QUERIES,
    ids=[
        "agi_research",
        "renewable_energy_stats",
        "rust_vs_cpp",
        "jwst_discoveries",
    ],
)
def test_researcher_selects_correct_tools(
    research_model, researcher_system_prompt, query, expectations
):
    """
    Given a research query, the LLM should select appropriate research tools
    and generate valid arguments.
    """
    t.log_inputs({"query": query, "expected_tools": expectations["expected_tools"]})

    # Make a single REAL LLM call
    messages = [
        SystemMessage(content=researcher_system_prompt),
        HumanMessage(content=query),
    ]
    response = research_model.invoke(messages)

    # Extract tool calls
    tool_calls = response.tool_calls if response.tool_calls else []
    tool_names = [tc["name"] for tc in tool_calls]
    tool_args = {tc["name"]: tc["args"] for tc in tool_calls}

    t.log_outputs({
        "tool_calls_made": tool_names,
        "tool_arguments": tool_args,
        "response_content": response.content[:300] if response.content else "",
    })

    # --- Assertions ---

    # 1. At least one tool should be called (researcher should use tools, not just answer)
    assert len(tool_calls) > 0, (
        f"Researcher should call at least one tool for query: {query}"
    )

    # 2. Only valid research tools should be used
    valid_tools = {"web_search", "fetch_webpage", "wikipedia_search"}
    for name in tool_names:
        assert name in valid_tools, (
            f"Unexpected tool '{name}' called. Expected one of {valid_tools}"
        )

    # 3. At least one of the expected tools should be called
    expected = set(expectations["expected_tools"])
    actual = set(tool_names)
    assert actual & expected, (
        f"Expected at least one of {expected} to be called, got {actual}"
    )

    # 4. Tool arguments should be well-formed
    for tc in tool_calls:
        if tc["name"] == "web_search":
            assert "query" in tc["args"], "web_search must have 'query' argument"
            assert isinstance(tc["args"]["query"], str), "web_search query must be a string"
            assert len(tc["args"]["query"]) > 0, "web_search query must not be empty"

        elif tc["name"] == "wikipedia_search":
            assert "query" in tc["args"], "wikipedia_search must have 'query' argument"
            assert isinstance(tc["args"]["query"], str), "wikipedia_search query must be a string"
            assert len(tc["args"]["query"]) > 0, "wikipedia_search query must not be empty"

        elif tc["name"] == "fetch_webpage":
            assert "url" in tc["args"], "fetch_webpage must have 'url' argument"
            assert tc["args"]["url"].startswith("http"), "fetch_webpage url must be a valid URL"

    # Log feedback scores
    t.log_feedback(key="tools_called", score=1 if len(tool_calls) > 0 else 0)
    t.log_feedback(
        key="correct_tool_selected",
        score=1 if actual & expected else 0,
    )
    t.log_feedback(key="only_valid_tools", score=1 if actual <= valid_tools else 0)
    t.log_feedback(
        key="args_well_formed",
        score=1,  # if we got here, all arg assertions passed
    )


# ---------------------------------------------------------------------------
# Test: web_search query relevance
# ---------------------------------------------------------------------------

@pytest.mark.langsmith(test_suite_name="Researcher Agent - Argument Quality")
@pytest.mark.parametrize(
    "query, expected_keyword",
    [
        ("Research quantum computing breakthroughs in 2025", "quantum"),
        ("What is the current GDP of India?", "GDP"),
        ("Find information about CRISPR gene editing technology", "CRISPR"),
    ],
    ids=["quantum_computing", "india_gdp", "crispr_gene_editing"],
)
def test_researcher_generates_relevant_search_queries(
    research_model, researcher_system_prompt, query, expected_keyword
):
    """
    The search query argument generated by the LLM should be relevant
    to the user's original query (contains key terms).
    """
    t.log_inputs({"query": query, "expected_keyword": expected_keyword})

    messages = [
        SystemMessage(content=researcher_system_prompt),
        HumanMessage(content=query),
    ]
    response = research_model.invoke(messages)

    tool_calls = response.tool_calls if response.tool_calls else []
    search_queries = [
        tc["args"].get("query", "")
        for tc in tool_calls
        if tc["name"] in ("web_search", "wikipedia_search")
    ]

    t.log_outputs({
        "search_queries_generated": search_queries,
        "tool_calls": [tc["name"] for tc in tool_calls],
    })

    # At least one search query should exist
    assert len(search_queries) > 0, "Should generate at least one search query"

    # At least one search query should contain the expected keyword (case-insensitive)
    keyword_found = any(
        expected_keyword.lower() in sq.lower() for sq in search_queries
    )
    assert keyword_found, (
        f"Expected keyword '{expected_keyword}' in at least one search query. "
        f"Got: {search_queries}"
    )

    t.log_feedback(key="has_search_queries", score=1)
    t.log_feedback(key="keyword_relevant", score=1 if keyword_found else 0)
