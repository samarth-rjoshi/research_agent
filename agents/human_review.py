

from langgraph.types import interrupt

from .state import AgentState


def human_review_node(state: AgentState) -> dict:
    """
    Human-in-the-loop review node.

    Shows the draft document and pauses for human feedback.
    Returns updated state with human_feedback.

    - "approve" / "ok" / "yes" → proceed to END
    - Anything else → feedback stored, routes back to Supervisor
    """
    draft = state.get("draft_document")

    print("\n" + "=" * 70)
    print("👤 HUMAN REVIEW")
    print("=" * 70)
    print("\n📄 Current Draft:\n")
    print(draft[:3000])
    if len(draft) > 3000:
        print(f"\n... [{len(draft) - 3000} more characters]")
    print("\n" + "-" * 70)

    # Pause and wait for human input
    human_input = interrupt(
        {
            "message": "Please review the draft above. Type 'approve' to finalize, or provide feedback for changes.",
            "draft_preview": draft[:500],
        }
    )

    feedback = human_input.strip() if isinstance(human_input, str) else str(human_input)

    # Check if approved
    if feedback.lower() in ("approve", "approved", "ok", "yes", "lgtm", "looks good"):
        print("✅ Draft APPROVED by human reviewer")
        return {
            "human_feedback": "",
            "current_phase": "approved",
        }
    else:
        print(f"📝 Human feedback received: {feedback[:100]}...")
        return {
            "human_feedback": feedback,
            "current_phase": "feedback",
        }
