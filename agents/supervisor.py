"""
Supervisor Agent

Analyzes prompts and human feedback to decide routing:
- "research" → fan out to parallel researchers
- "rewrite" → send directly to writer
"""

from langchain_core.messages import HumanMessage, SystemMessage

from .state import AgentState
from prompts import load_prompt
from utils import get_llm


async def run_supervisor(state: AgentState) -> dict:
    """
    Execute the supervisor.

    Analyzes the current state and decides whether to research or rewrite.
    """
    from .models import SupervisorPlan

    print("\n🧠 SUPERVISOR Starting...")

    model = get_llm(temperature=0.1)

    human_feedback = state.get("human_feedback", "")
    existing_draft = state.get("draft_document", "")

    # Always extract the original query for context
    original_query = ""
    for msg in state.get("messages", []):
        if isinstance(msg, HumanMessage):
            original_query = msg.content
            break

    if human_feedback:
        # Feedback loop — decide based on human feedback
        user_content = f"""The human reviewed the current draft and provided this feedback:

ORIGINAL QUERY: {original_query}

FEEDBACK: {human_feedback}

CURRENT DRAFT (first 2000 chars):
{existing_draft[:2000]}

Decide whether this feedback requires more research or just a rewrite of the existing draft.
If research is needed, create exactly 2 focused subtopics related to the ORIGINAL QUERY and the feedback."""
    else:
        # Initial query
        user_content = f"""Plan the research for this query:

QUERY: {original_query}

Break this into exactly 2 focused subtopics for parallel research."""

    system_prompt = load_prompt("supervisor_system")
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_content),
    ]

    # Use structured output
    structured_llm = model.with_structured_output(SupervisorPlan)
    plan: SupervisorPlan = structured_llm.invoke(messages)

    action = plan.action
    subtopics = plan.subtopics or []
    rewrite_instructions = plan.rewrite_instructions or ""

    print(f"   📋 Action: {action}")
    if action == "research":
        print(f"   📚 Subtopics: {subtopics}")
    else:
        print(f"   ✏️  Rewrite instructions: {rewrite_instructions[:100]}...")

    result = {
        "current_phase": action,
        "rewrite_instructions": rewrite_instructions
    }

    if action == "research":
        result["subtopics"] = subtopics

    return result
