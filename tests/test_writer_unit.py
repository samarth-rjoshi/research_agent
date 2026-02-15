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
    with patch("agents.base.ChatOpenAI"):
        agent = WriterAgent(tools=mock_tools)
        return agent


@pytest.mark.asyncio
async def test_writer_run_initial_draft(writer_agent):
    """Test writer agent initial draft mode."""
    final_messages = [
        HumanMessage(content="Based on the following research data..."),
        AIMessage(content="Initial Draft Content"),
    ]
    mock_agent = AsyncMock()
    mock_agent.ainvoke.return_value = {"messages": final_messages}

    with patch("agents.writer.create_agent", return_value=mock_agent):
        state: AgentState = {
            "messages": [HumanMessage(content="Write a report about AI")],
            "research_data": "AI is growing.",
            "draft_document": "",
            "current_phase": "writing",
        }

        result = await writer_agent.run(state)

    assert result["draft_document"] == "Initial Draft Content"
    assert result["current_phase"] == "human_review"
    assert "messages" in result

    # Verify prompt contents (roughly)
    call_args = mock_agent.ainvoke.call_args[0][0]
    prompt_msg = call_args["messages"][0]
    assert "Based on the following research data" in prompt_msg.content
    assert "AI is growing." in prompt_msg.content


@pytest.mark.asyncio
async def test_writer_run_revision(writer_agent):
    """Test writer agent revision mode."""
    final_messages = [
        HumanMessage(content="Revise the following document..."),
        AIMessage(content="Revised Draft Content"),
    ]
    mock_agent = AsyncMock()
    mock_agent.ainvoke.return_value = {"messages": final_messages}

    with patch("agents.writer.create_agent", return_value=mock_agent):
        state: AgentState = {
            "messages": [HumanMessage(content="Write a report about AI")],
            "research_data": "AI is growing.",
            "draft_document": "Initial draft content",
            "rewrite_instructions": "Add more about ethics.",
            "current_phase": "writing",
        }

        result = await writer_agent.run(state)

    assert result["draft_document"] == "Revised Draft Content"
    assert result["rewrite_instructions"] == ""  # Should be cleared

    # Verify prompt contents (roughly)
    call_args = mock_agent.ainvoke.call_args[0][0]
    prompt_msg = call_args["messages"][0]
    assert "Revise the following document" in prompt_msg.content
    assert "Add more about ethics." in prompt_msg.content
    assert "Initial draft content" in prompt_msg.content


@pytest.mark.asyncio
async def test_writer_run_with_tools(writer_agent):
    """Test writer agent with tool calls (e.g. write_document)."""
    tool_call = {"name": "write_document", "args": {"filename": "test.md", "content": "data"}, "id": "call_1"}
    final_messages = [
        HumanMessage(content="Based on the following research data..."),
        AIMessage(content="", tool_calls=[tool_call]),
        ToolMessage(content="File written", tool_call_id="call_1"),
        AIMessage(content="Final response"),
    ]
    mock_agent = AsyncMock()
    mock_agent.ainvoke.return_value = {"messages": final_messages}

    with patch("agents.writer.create_agent", return_value=mock_agent):
        state: AgentState = {
            "messages": [HumanMessage(content="Write a report")],
            "research_data": "Some data",
            "draft_document": "",
            "current_phase": "writing",
        }

        result = await writer_agent.run(state)

    assert result["draft_document"] == "Final response"
    assert len(result["messages"]) == 4
