"""
Supervisor Agent

Analyzes prompts and human feedback to decide routing:
- "research" → fan out to parallel researchers
- "rewrite" → send directly to writer
"""

import json
from typing import List
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

from .base import BaseAgent
from .state import AgentState


class SupervisorAgent(BaseAgent):
    """
    Supervisor Agent - Central brain that routes work to the right agents.

    On first call: analyzes the user's query and breaks it into subtopics for research.
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
        """
        Execute the supervisor agent.

        Analyzes the current state and decides whether to research or rewrite.
        """
        from .models import SupervisorPlan
        
        print("\n🧠 SUPERVISOR AGENT Starting...")

        human_feedback = state.get("human_feedback", "")
        existing_draft = state.get("draft_document", "")

        if human_feedback:
            # Feedback loop — decide based on human feedback
            user_content = f"""The human reviewed the current draft and provided this feedback:

FEEDBACK: {human_feedback}

CURRENT DRAFT (first 2000 chars):
{existing_draft[:2000]}

Decide whether this feedback requires more research or just a rewrite of the existing draft."""
        else:
            # Initial query
            original_query = ""
            for msg in state.get("messages", []):
                if isinstance(msg, HumanMessage):
                    original_query = msg.content
                    break

            user_content = f"""Plan the research for this query:

QUERY: {original_query}

Break this into 3-5 focused subtopics for parallel research."""

        messages = [
            self.get_system_message(),
            HumanMessage(content=user_content),
        ]

        # Use structured output
        structured_llm = self.model.with_structured_output(SupervisorPlan)
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
