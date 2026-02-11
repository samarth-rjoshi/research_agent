import pytest
from unittest.mock import patch, MagicMock
from agents.human_review import human_review_node
from agents.state import AgentState

def test_human_review_node_approve():
    """Test human review node when approved."""
    state: AgentState = {
        "draft_document": "This is a great document.",
        "current_phase": "human_review"
    }
    
    with patch("agents.human_review.interrupt", return_value="approve") as mock_interrupt:
        result = human_review_node(state)
        
        assert result["current_phase"] == "approved"
        assert result["human_feedback"] == ""
        mock_interrupt.assert_called_once()

def test_human_review_node_feedback():
    """Test human review node when feedback is provided."""
    state: AgentState = {
        "draft_document": "This is a document.",
        "current_phase": "human_review"
    }
    
    with patch("agents.human_review.interrupt", return_value="Please add more details.") as mock_interrupt:
        result = human_review_node(state)
        
        assert result["current_phase"] == "feedback"
        assert result["human_feedback"] == "Please add more details."
        mock_interrupt.assert_called_once()

def test_human_review_node_approve_variations():
    """Test various approval keywords."""
    state: AgentState = {"draft_document": "doc", "current_phase": "hr"}
    
    for ok_msg in ["approve", "LGTM", "ok", "yes"]:
        with patch("agents.human_review.interrupt", return_value=ok_msg):
            result = human_review_node(state)
            assert result["current_phase"] == "approved"
