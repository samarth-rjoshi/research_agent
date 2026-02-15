import pytest
from unittest.mock import MagicMock, patch, AsyncMock
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage, SystemMessage
from agents.researcher import run_researcher
from agents.state import AgentState


@pytest.mark.asyncio
async def test_researcher_run_no_tools():
    """Test researcher when the model returns a direct answer (no tool calls)."""
    mock_response = AIMessage(content="Search result summary")

    mock_model_instance = MagicMock()
    mock_model_with_tools = MagicMock()
    mock_model_with_tools.ainvoke = AsyncMock(return_value=mock_response)
    mock_model_instance.bind_tools.return_value = mock_model_with_tools

    mock_tools = [MagicMock(name="web_search")]

    with patch("agents.researcher.get_llm", return_value=mock_model_instance):
        state: AgentState = {
            "messages": [HumanMessage(content="Test query")],
            "research_data": "",
            "draft_document": "",
            "current_phase": "research",
        }

        result = await run_researcher(state, tools=mock_tools)

    assert result["parallel_results"] == ["Search result summary"]
    assert "messages" in result


@pytest.mark.asyncio
async def test_researcher_run_with_tools():
    """Test researcher when tools are called during the loop."""
    tool_call = {"name": "web_search", "args": {"query": "AI"}, "id": "call_1"}

    # First response has a tool call, second is the final answer
    response_with_tool = AIMessage(content="", tool_calls=[tool_call])
    response_final = AIMessage(content="Final research summary")

    mock_model_instance = MagicMock()
    mock_model_with_tools = MagicMock()
    mock_model_with_tools.ainvoke = AsyncMock(
        side_effect=[response_with_tool, response_final]
    )
    mock_model_instance.bind_tools.return_value = mock_model_with_tools

    mock_tool = MagicMock()
    mock_tool.name = "web_search"
    mock_tool.ainvoke = AsyncMock(return_value="Tool output")

    with patch("agents.researcher.get_llm", return_value=mock_model_instance):
        state: AgentState = {
            "messages": [HumanMessage(content="Test query")],
            "research_data": "",
            "draft_document": "",
            "current_phase": "research",
        }

        result = await run_researcher(state, tools=[mock_tool])

    assert result["parallel_results"] == ["Final research summary"]
    assert "messages" in result
