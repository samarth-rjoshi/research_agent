import pytest
from unittest.mock import MagicMock, patch, AsyncMock
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage, SystemMessage
from agents.researcher import ResearcherAgent
from agents.state import AgentState


@pytest.fixture
def mock_tools():
    return [MagicMock(name="web_search")]


@pytest.fixture
def researcher_agent(mock_tools):
    with patch("agents.base.ChatOpenAI"):
        agent = ResearcherAgent(tools=mock_tools)
        return agent


@pytest.mark.asyncio
async def test_researcher_run_no_tools(researcher_agent):
    """Test researcher agent when the inner agent returns a direct answer."""
    final_messages = [
        HumanMessage(content="Test query"),
        AIMessage(content="Search result summary"),
    ]
    mock_agent = AsyncMock()
    mock_agent.ainvoke.return_value = {"messages": final_messages}

    with patch("agents.researcher.create_agent", return_value=mock_agent):
        state: AgentState = {
            "messages": [HumanMessage(content="Test query")],
            "research_data": "",
            "draft_document": "",
            "current_phase": "research",
        }

        result = await researcher_agent.run(state)

    assert result["current_phase"] == "writing"
    assert result["parallel_results"] == ["Search result summary"]
    assert result["messages"] == final_messages


@pytest.mark.asyncio
async def test_researcher_run_with_tools(researcher_agent):
    """Test researcher agent when tools were called during the loop."""
    tool_call = {"name": "web_search", "args": {"query": "AI"}, "id": "call_1"}
    final_messages = [
        HumanMessage(content="Test query"),
        AIMessage(content="", tool_calls=[tool_call]),
        ToolMessage(content="Tool output", tool_call_id="call_1"),
        AIMessage(content="Final research summary"),
    ]
    mock_agent = AsyncMock()
    mock_agent.ainvoke.return_value = {"messages": final_messages}

    with patch("agents.researcher.create_agent", return_value=mock_agent):
        state: AgentState = {
            "messages": [HumanMessage(content="Test query")],
            "research_data": "",
            "draft_document": "",
            "current_phase": "research",
        }

        result = await researcher_agent.run(state)

    assert result["current_phase"] == "writing"
    assert result["parallel_results"] == ["Final research summary"]
    # All messages from the agent run are returned
    assert len(result["messages"]) == 4
