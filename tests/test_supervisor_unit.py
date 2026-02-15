import pytest
from unittest.mock import MagicMock, patch
from langchain_core.messages import HumanMessage
from agents.supervisor import run_supervisor
from agents.models import SupervisorPlan
from agents.state import AgentState


@pytest.mark.asyncio
async def test_supervisor_run_initial_query():
    """Test supervisor initial query mode (research path)."""
    mock_plan = SupervisorPlan(
        action="research",
        subtopics=["topic1", "topic2"],
        rewrite_instructions=None,
    )

    mock_structured_llm = MagicMock()
    mock_structured_llm.invoke.return_value = mock_plan

    mock_model_instance = MagicMock()
    mock_model_instance.with_structured_output.return_value = mock_structured_llm

    with patch("agents.supervisor.get_llm", return_value=mock_model_instance):
        state: AgentState = {
            "messages": [HumanMessage(content="Explain quantum physics")],
            "current_phase": "start",
        }

        result = await run_supervisor(state)

    assert result["current_phase"] == "research"
    assert result["subtopics"] == ["topic1", "topic2"]
    assert result["rewrite_instructions"] == ""

    # Check if correct user content was generated
    prompt_msg = mock_structured_llm.invoke.call_args[0][0][1]
    assert "Explain quantum physics" in prompt_msg.content
    assert "Plan the research" in prompt_msg.content


@pytest.mark.asyncio
async def test_supervisor_run_feedback_loop():
    """Test supervisor feedback loop mode (rewrite path)."""
    mock_plan = SupervisorPlan(
        action="rewrite",
        subtopics=None,
        rewrite_instructions="Fix the introduction.",
    )

    mock_structured_llm = MagicMock()
    mock_structured_llm.invoke.return_value = mock_plan

    mock_model_instance = MagicMock()
    mock_model_instance.with_structured_output.return_value = mock_structured_llm

    with patch("agents.supervisor.get_llm", return_value=mock_model_instance):
        state: AgentState = {
            "messages": [HumanMessage(content="Explain quantum physics")],
            "human_feedback": "The intro is weak.",
            "draft_document": "Quantum physics is cool.",
            "current_phase": "human_review",
        }

        result = await run_supervisor(state)

    assert result["current_phase"] == "rewrite"
    assert "subtopics" not in result
    assert result["rewrite_instructions"] == "Fix the introduction."

    # Check if feedback was included in prompt
    prompt_msg = mock_structured_llm.invoke.call_args[0][0][1]
    assert "The intro is weak." in prompt_msg.content
    assert "Quantum physics is cool." in prompt_msg.content
