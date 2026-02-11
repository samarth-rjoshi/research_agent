"""
Supervisor Agent

Analyzes prompts and human feedback to decide routing:
- "research" → fan out to parallel researchers
- "rewrite" → send directly to writer
"""

from langchain_core.messages import HumanMessage

from .base import BaseAgent
from .state import AgentState
from .models import SupervisorPlan


class SupervisorAgent(BaseAgent):
    """
    Supervisor Agent - Central brain that routes work to the right agents.

    On first call: analyzes the user's query and breaks it into subtopics.
    On feedback loops: reads human_feedback and decides whether to research more or rewrite.
    """

    def __init__(self, model_name: str = "gpt-4o", temperature: float = 0.1):
        super().__init__(tools=[], model_name=model_name, temperature=temperature)

    @property
    def name(self) -> str:
        return "Supervisor"

    @property
    def system_prompt(self) -> str:
        return self.load_prompt("supervisor_system")

    async def run(self, state: AgentState) -> dict:
        print("\n🧠 SUPERVISOR AGENT Starting...")

        human_feedback = state.get("human_feedback", "")
        existing_draft = state.get("draft_document", "")

        # Build prompt content
        if human_feedback:
            user_content = f"""The human reviewed the current draft and provided this feedback:

FEEDBACK: {human_feedback}

CURRENT DRAFT (first 2000 chars):
{existing_draft[:2000]}

Decide whether this feedback requires more research or just a rewrite.
"""
        else:
            original_query = next(
                (msg.content for msg in state.get("messages", [])
                 if isinstance(msg, HumanMessage)),
                ""
            )

            user_content = f"""Plan the research for this query:

QUERY: {original_query}

Break this into 3-5 focused subtopics for parallel research.
"""

        messages = [
            self.get_system_message(),
            HumanMessage(content=user_content),
        ]

        # ✅ Modern structured output pattern
        structured_llm = self.model.with_structured_output(SupervisorPlan)
        plan: SupervisorPlan = await structured_llm.ainvoke(messages)

        print(f"   📋 Action: {plan.action}")

        if plan.action == "research":
            print(f"   📚 Subtopics: {plan.subtopics}")
        else:
            print(f"   ✏️  Rewrite instructions: {plan.rewrite_instructions[:100]}...")

        result = {
            "current_phase": plan.action,
            "rewrite_instructions": plan.rewrite_instructions or ""
        }

        if plan.action == "research":
            result["subtopics"] = plan.subtopics or []

        return result
