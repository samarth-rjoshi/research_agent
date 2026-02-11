import pytest
from unittest.mock import MagicMock, patch
from langchain_core.messages import HumanMessage
from agents.supervisor import SupervisorAgent
from agents.models import SupervisorPlan
from agents.state import AgentState

@pytest.fixture
def supervisor_agent():
    with patch("agents.base.BaseAgent.load_prompt", return_value="Mock system prompt"):
        with patch("agents.base.ChatOpenAI"):
            agent = SupervisorAgent()
            return agent

@pytest.mark.asyncio
async def test_supervisor_run_initial_query(supervisor_agent):
    """Test supervisor agent initial query mode (research path)."""
    mock_plan = SupervisorPlan(
        action="research",
        subtopics=["topic1", "topic2"],
        rewrite_instructions=None
    )
    
    # Mock model.with_structured_output().invoke()
    mock_structured_llm = MagicMock()
    mock_structured_llm.invoke.return_value = mock_plan
    supervisor_agent.model.with_structured_output.return_value = mock_structured_llm
    
    state: AgentState = {
        "messages": [HumanMessage(content="Explain quantum physics")],
        "current_phase": "start"
    }
    
    result = await supervisor_agent.run(state)
    
    assert result["current_phase"] == "research"
    assert result["subtopics"] == ["topic1", "topic2"]
    assert result["rewrite_instructions"] == ""
    
    # Check if correct user content was generated
    prompt_msg = supervisor_agent.model.with_structured_output.return_value.invoke.call_args[0][0][1]
    assert "Explain quantum physics" in prompt_msg.content
    assert "Plan the research" in prompt_msg.content

@pytest.mark.asyncio
async def test_supervisor_run_feedback_loop(supervisor_agent):
    """Test supervisor agent feedback loop mode (rewrite path)."""
    mock_plan = SupervisorPlan(
        action="rewrite",
        subtopics=None,
        rewrite_instructions="Fix the introduction."
    )
    
    mock_structured_llm = MagicMock()
    mock_structured_llm.invoke.return_value = mock_plan
    supervisor_agent.model.with_structured_output.return_value = mock_structured_llm
    
    state: AgentState = {
        "messages": [HumanMessage(content="Explain quantum physics")],
        "human_feedback": "The intro is weak.",
        "draft_document": "Quantum physics is cool.",
        "current_phase": "human_review"
    }
    
    result = await supervisor_agent.run(state)
    
    assert result["current_phase"] == "rewrite"
    assert "subtopics" not in result
    assert result["rewrite_instructions"] == "Fix the introduction."
    
    # Check if feedback was included in prompt
    prompt_msg = supervisor_agent.model.with_structured_output.return_value.invoke.call_args[0][0][1]
    assert "The intro is weak." in prompt_msg.content
    assert "Quantum physics is cool." in prompt_msg.content
