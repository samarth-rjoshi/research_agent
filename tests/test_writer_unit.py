import pytest
from unittest.mock import MagicMock, patch, AsyncMock
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from agents.writer import WriterAgent
from agents.state import AgentState

@pytest.fixture
def mock_tools():
    return [MagicMock(name="write_document")]

@pytest.fixture
def writer_agent(mock_tools):
    with patch("agents.base.BaseAgent.load_prompt", return_value="Mock system prompt"):
        with patch("agents.base.ChatOpenAI"):
            agent = WriterAgent(tools=mock_tools)
            return agent

@pytest.mark.asyncio
async def test_writer_run_initial_draft(writer_agent):
    """Test writer agent initial draft mode."""
    mock_response = AIMessage(content="Initial Draft Content")
    writer_agent.model_with_tools.invoke = MagicMock(return_value=mock_response)
    
    state: AgentState = {
        "messages": [HumanMessage(content="Write a report about AI")],
        "research_data": "AI is growing.",
        "draft_document": "",
        "current_phase": "writing"
    }
    
    result = await writer_agent.run(state)
    
    assert result["draft_document"] == "Initial Draft Content"
    assert result["current_phase"] == "human_review"
    assert "messages" in result
    
    # Verify prompt contents (roughly)
    writer_agent.model_with_tools.invoke.assert_called_once()
    prompt_msg = writer_agent.model_with_tools.invoke.call_args[0][0][1]
    assert "Based on the following research data" in prompt_msg.content
    assert "AI is growing." in prompt_msg.content

@pytest.mark.asyncio
async def test_writer_run_revision(writer_agent):
    """Test writer agent revision mode."""
    mock_response = AIMessage(content="Revised Draft Content")
    writer_agent.model_with_tools.invoke = MagicMock(return_value=mock_response)
    
    state: AgentState = {
        "messages": [HumanMessage(content="Write a report about AI")],
        "research_data": "AI is growing.",
        "draft_document": "Initial draft content",
        "rewrite_instructions": "Add more about ethics.",
        "current_phase": "writing"
    }
    
    result = await writer_agent.run(state)
    
    assert result["draft_document"] == "Revised Draft Content"
    assert result["rewrite_instructions"] == ""  # Should be cleared
    
    # Verify prompt contents (roughly)
    writer_agent.model_with_tools.invoke.assert_called_once()
    prompt_msg = writer_agent.model_with_tools.invoke.call_args[0][0][1]
    assert "Revise the following document" in prompt_msg.content
    assert "Add more about ethics." in prompt_msg.content
    assert "Initial draft content" in prompt_msg.content

@pytest.mark.asyncio
async def test_writer_run_with_tools(writer_agent):
    """Test writer agent with tool calls (e.g. write_document)."""
    tool_call = {"name": "write_document", "args": {"filename": "test.md", "content": "data"}, "id": "call_1"}
    response_1 = AIMessage(content="", tool_calls=[tool_call])
    response_2 = AIMessage(content="Final response")
    
    writer_agent.model_with_tools.invoke = MagicMock(side_effect=[response_1, response_2])
    
    mock_tool_node = AsyncMock()
    mock_tool_node.ainvoke.return_value = {
        "messages": [ToolMessage(content="File written", tool_call_id="call_1")]
    }
    
    state: AgentState = {
        "messages": [HumanMessage(content="Write a report")],
        "research_data": "Some data",
        "draft_document": "",
        "current_phase": "writing"
    }
    
    with patch("langgraph.prebuilt.ToolNode", return_value=mock_tool_node):
        result = await writer_agent.run(state)
    
    assert result["draft_document"] == "Final response"
    assert len(result["messages"]) == 3
