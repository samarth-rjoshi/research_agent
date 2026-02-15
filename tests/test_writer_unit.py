import pytest
from unittest.mock import MagicMock, patch, AsyncMock
from langchain_core.messages import AIMessage, HumanMessage
from agents.writer import run_writer
from agents.state import AgentState


@pytest.mark.asyncio
async def test_writer_run_initial_draft():
    """Test writer initial draft mode."""
    mock_response = AIMessage(content="Initial Draft Content")

    mock_model_instance = MagicMock()
    mock_model_instance.ainvoke = AsyncMock(return_value=mock_response)

    with patch("agents.writer.get_llm", return_value=mock_model_instance):
        
        state: AgentState = {
            "messages": [HumanMessage(content="Write a report about AI")],
            "research_data": "AI is growing.",
            "draft_document": "",
            "current_phase": "writing",
        }

        result = await run_writer(state)

    assert result["draft_document"] == "Initial Draft Content"
    assert result["current_phase"] == "human_review"
    assert "messages" in result


@pytest.mark.asyncio
async def test_writer_run_revision():
    """Test writer revision mode."""
    mock_response = AIMessage(content="Revised Draft Content")

    mock_model_instance = MagicMock()
    mock_model_instance.ainvoke = AsyncMock(return_value=mock_response)

    with patch("agents.writer.get_llm", return_value=mock_model_instance):
        
        state: AgentState = {
            "messages": [HumanMessage(content="Write a report about AI")],
            "research_data": "AI is growing.",
            "draft_document": "Initial draft content",
            "rewrite_instructions": "Add more about ethics.",
            "current_phase": "writing",
        }

        result = await run_writer(state)

    assert result["draft_document"] == "Revised Draft Content"
    assert result["rewrite_instructions"] == ""  # Should be cleared
