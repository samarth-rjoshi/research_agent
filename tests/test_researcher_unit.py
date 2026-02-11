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
    with patch("agents.base.BaseAgent.load_prompt", return_value="Mock system prompt"):
        with patch("agents.base.ChatOpenAI"):
            agent = ResearcherAgent(tools=mock_tools)
            return agent

@pytest.mark.asyncio
async def test_researcher_run_no_tools(researcher_agent):
    """Test researcher agent when it returns a direct answer without tool calls."""
    mock_response = AIMessage(content="Search result summary")
    researcher_agent.model_with_tools.invoke = MagicMock(return_value=mock_response)
    
    state: AgentState = {
        "messages": [HumanMessage(content="Test query")],
        "research_data": "",
        "draft_document": "",
        "current_phase": "research"
    }
    
    result = await researcher_agent.run(state)
    
    assert result["current_phase"] == "writing"
    assert result["parallel_results"] == ["Search result summary"]
    assert len(result["messages"]) == 1
    assert isinstance(result["messages"][0], AIMessage)
    assert result["messages"][0].content == "Search result summary"

@pytest.mark.asyncio
async def test_researcher_run_with_tools(researcher_agent):
    """Test researcher agent with tool calls in the loop."""
    # 1. First call returns tool call
    # 2. Second call returns final content
    tool_call = {"name": "web_search", "args": {"query": "AI"}, "id": "call_1"}
    response_1 = AIMessage(content="", tool_calls=[tool_call])
    response_2 = AIMessage(content="Final research summary")
    
    researcher_agent.model_with_tools.invoke = MagicMock(side_effect=[response_1, response_2])
    
    # Mock ToolNode
    mock_tool_node = AsyncMock()
    mock_tool_node.ainvoke.return_value = {
        "messages": [ToolMessage(content="Tool output", tool_call_id="call_1")]
    }
    
    state: AgentState = {
        "messages": [HumanMessage(content="Test query")],
        "research_data": "",
        "draft_document": "",
        "current_phase": "research"
    }
    
    with patch("langgraph.prebuilt.ToolNode", return_value=mock_tool_node):
        result = await researcher_agent.run(state)
    
    assert result["current_phase"] == "writing"
    assert result["parallel_results"] == ["Final research summary"]
    # Messages: AIMessage (tool call) + ToolMessage (result) + AIMessage (final)
    assert len(result["messages"]) == 3
    assert result["messages"][0].tool_calls == [tool_call]
    assert isinstance(result["messages"][1], ToolMessage)
    assert result["messages"][2].content == "Final research summary"
