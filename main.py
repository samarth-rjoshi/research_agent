"""
Multi-Agent Web Research and Document Writer

A LangGraph-based multi-agent system with a Supervisor-led dynamic architecture:
- Supervisor: Routes to Parallel Researchers or Writer
- Researchers: Run in parallel based on subtopics
- Writer: Synthesizes research or revises based on feedback
- Human Review: Mandatory checkpoint after writing
"""

import asyncio
import os
import time
import uuid
from typing import List, Literal
from dotenv import load_dotenv

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Send, Command
from langchain_core.messages import HumanMessage

from tools import get_mcp_client
from agents import (
    run_supervisor,
    run_researcher,
    run_writer,
    human_review_node,
    AgentState
)

# Load environment variables
load_dotenv()


# --- Node Functions ---

async def supervisor_node(state: AgentState):
    """Router node that calls the supervisor."""
    return await run_supervisor(state)


async def researcher_node(state: AgentState):
    """Researcher node (fanned out)."""
    #TODO: FETCHING THE TOOLS TWICE 
    from tools import get_tools
    all_tools = await get_tools()
    research_tools = [t for t in all_tools if t.name in ["web_search", "fetch_webpage", "wikipedia_search"]]

    return await run_researcher(state, tools=research_tools)


async def merge_research_node(state: AgentState):
    """Merges parallel research results into a single research_data string."""
    print("🔄 MERGING parallel research results...")
    results = state.get("parallel_results", [])
    existing_research = state.get("research_data", "")
    merged = ""
    
    for i, res in enumerate(results, 1):
        merged += f"--- SOURCE {i} ---\n{res}\n\n"
    
    # Append new research to existing data so previous rounds aren't lost
    if existing_research:
        combined = existing_research + "\n--- ADDITIONAL RESEARCH ---\n\n" + merged
    else:
        combined = merged

    return {
        "research_data": combined,
        "parallel_results": [], 
        "current_phase": "writing"
    }


async def writer_node(state: AgentState):
    """Writer node."""
    return await run_writer(state)


# --- Routing Functions ---

def route_from_supervisor(state: AgentState):
    """Decides where to go after Supervisor based on current_phase."""
    phase = state.get("current_phase")
    
    if phase == "research":
        subtopics = state.get("subtopics", [])
        
        # Parallel fan-out
        return [Send("researcher", {"messages": [HumanMessage(content=s)]}) for s in subtopics]
    
    elif phase == "rewrite":
        return "writer"
    
    return "writer" # Fallback


def route_from_human_review(state: AgentState):
    """Decides where to go after Human Review."""
    phase = state.get("current_phase")
    if phase == "approved":
        return END
    return "supervisor"


# --- Graph Construction ---

def create_multi_agent_graph():
    """Build and compile the multi-agent graph."""
    builder = StateGraph(AgentState)
    
    # Add nodes
    builder.add_node("supervisor", supervisor_node)
    builder.add_node("researcher", researcher_node)
    builder.add_node("merge_research", merge_research_node)
    builder.add_node("writer", writer_node)
    builder.add_node("human_review", human_review_node)
    
    # Define flow
    builder.add_edge(START, "supervisor")
    
    # Supervisor → Parallel Researchers OR Writer
    builder.add_conditional_edges(
        "supervisor", 
        route_from_supervisor,
        ["researcher", "writer"]
    )
    
    # Parallel Researchers → Merge
    builder.add_edge("researcher", "merge_research")
    
    # Merge → Writer
    builder.add_edge("merge_research", "writer")
    
    # Writer → Human Review (Checkpoint)
    builder.add_edge("writer", "human_review")
    
    # Human Review → END or Back to Supervisor
    builder.add_conditional_edges(
        "human_review",
        route_from_human_review,
        [END, "supervisor"]
    )
    
    # Persistence for interrupts
    memory = MemorySaver()
    return builder.compile(checkpointer=memory)


async def run_multi_agent(query: str):
    """
    Run the multi-agent pipeline with a given research query.
    Handles the interrupt-resume loop for human review.
    """
    start_time = time.time()
    thread_id = uuid.uuid4().hex[:8]

    print("\n" + "=" * 70)
    print("🤖 Supervisor-Led Multi-Agent Pipeline")
    print("=" * 70)
    print(f"\n📝 Task: {query}")
    print(f"🧵 Thread: {thread_id}\n")
    
    graph = create_multi_agent_graph()
    config = {"configurable": {"thread_id": thread_id}}
    
    # Initialize state
    initial_state = {
        "messages": [HumanMessage(content=query)],
        "research_data": "",
        "parallel_results": [],
        "draft_document": "",
        "human_feedback": "",
        "rewrite_instructions": "",
        "subtopics": [],
        "current_phase": "initial"
    }
    
    # Event loop to handle interrupts
    current_input = initial_state
    
    while True:
        async for event in graph.astream(current_input, config, stream_mode="values"):
            # Values stream will emit the latest state at each step
            latest_state = event
        
        # Check if we are at an interrupt
        state_snapshot = await graph.aget_state(config)
        
        if state_snapshot.next:
            # We hit an interrupt (human_review node)
            # The human_review_node already printed the draft.
            print("\n👉 Awaiting your input (type 'approve' to finish, or describe changes):")
            user_input = await asyncio.get_event_loop().run_in_executor(None, input, "Feedback > ")
            
            # Resume with user input
            current_input = Command(resume=user_input)
        else:
            # Graph finished
            break
    
    elapsed = time.time() - start_time
    print("\n" + "=" * 70)
    print(f"📊 PIPELINE COMPLETE  ⏱  {elapsed:.1f}s")
    print("=" * 70)
    
    return latest_state


async def main():
    """Interactive entry point — prompts the user for research topics in a loop."""
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ Error: OPENAI_API_KEY environment variable not set")
        return

    print("\n" + "=" * 70)
    print("  🔬  Multi-Agent Research Assistant")
    print("=" * 70)
    print("  Describe a topic and I'll research, write, and let you review.")
    print("  Type 'quit' to exit.\n")

    while True:
        try:
            query = await asyncio.get_event_loop().run_in_executor(
                None, input, "🔎 Enter research topic > "
            )
        except (EOFError, KeyboardInterrupt):
            print("\n👋 Goodbye!")
            break

        query = query.strip()
        if not query:
            continue
        if query.lower() in ("quit", "exit", "q"):
            print("👋 Goodbye!")
            break

        try:
            await run_multi_agent(query)
        except Exception as exc:
            print(f"\n❌ Pipeline error: {exc}")
            print("   You can try again with another topic.\n")


if __name__ == "__main__":
    asyncio.run(main())
