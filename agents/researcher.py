"""
Researcher Agent

Gathers information from web sources, Wikipedia, and URLs.
Uses a ToolNode-based loop for tool execution.
"""

import os

from langchain_core.messages import SystemMessage
from langgraph.prebuilt import ToolNode

from .state import AgentState
from prompts import load_prompt
from utils import get_llm


async def run_researcher(state: AgentState, tools: list) -> dict:
    """
    Execute the researcher with a ToolNode-based loop.

    Args:
        state: Current agent state
        tools: List of research tools (web_search, fetch_webpage, wikipedia_search)

    Returns:
        Updated state with research_data
    """
    print("\n🔍 RESEARCHER Starting...")

    model = get_llm(temperature=0.1)

    system_prompt = load_prompt("researcher")
    model_with_tools = model.bind_tools(tools)
    tool_node = ToolNode(tools)

    messages = [
        SystemMessage(content=system_prompt),
        *list(state["messages"]),
    ]

    # ToolNode-based loop
    iterations = 0
    max_iterations = int(os.getenv("RESEARCHER_MAX_ITERATIONS", "5"))
    while iterations < max_iterations:
        iterations += 1
        response = await model_with_tools.ainvoke(messages)
        messages.append(response)

        # No tool calls → model is done
        if not response.tool_calls:
            break

        # Execute tool calls via ToolNode
        tool_result = await tool_node.ainvoke({"messages": messages})
        messages.extend(tool_result["messages"])

    # Extract research data from the final AI message
    research_data = ""
    for msg in reversed(messages):
        if hasattr(msg, "content") and msg.content and not getattr(msg, "tool_calls", None):
            research_data = msg.content
            break

    print(f"✅ RESEARCHER Complete - Gathered {len(research_data)} chars of research")

    return {
        "messages": messages,
        "parallel_results": [research_data],
    }
