from langgraph.types import interrupt
from langchain_core.messages import HumanMessage

from .state import AgentState
from utils.pdf import write_pdf


def human_review_node(state: AgentState) -> dict:
    """
    Human-in-the-loop review node.

    Shows the draft document and pauses for human feedback.
    Returns updated state with human_feedback.

    - "approve" / "ok" / "yes" → proceed to END, save PDF
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

        # Extract query for filename
        original_query = ""
        for msg in state.get("messages", []):
            if isinstance(msg, HumanMessage):
                original_query = msg.content
                break
        
        filename = "report.pdf"
        if original_query:
            clean_query = "".join(c for c in original_query[:30] if c.isalnum() or c == " ").strip()
            if clean_query:
                filename = f"{clean_query.replace(' ', '_')}.pdf"

        # Save to PDF locally
        print(f"📄 Saving FINAL document to {filename}...")
        pdf_result = write_pdf(filename, draft)
        print(f"ℹ️ {pdf_result}")

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
