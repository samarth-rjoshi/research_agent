"""
Multi-Agent Web Research and Document Writer

A LangGraph-based multi-agent system with specialized agents:
- Researcher Agent: Gathers information from web sources
- Writer Agent: Synthesizes research into documents
- Reviewer Agent: Fact-checks and improves documents

Uses MCP servers for web research and document writing tools.
"""

import asyncio
import os
from dotenv import load_dotenv

from langgraph.graph import StateGraph, START, END
from langchain_core.messages import HumanMessage

from tools import get_mcp_client
from agents import ResearcherAgent, WriterAgent, ReviewerAgent, AgentState

# Load environment variables
load_dotenv()


async def create_multi_agent_graph():
    """
    Create and compile the multi-agent LangGraph.
    
    Returns:
        Compiled LangGraph with Researcher → Writer → Reviewer pipeline
    """
    # Get MCP client and tools
    client = get_mcp_client()
    all_tools = await client.get_tools()
    
    # Separate tools by category
    research_tools = [t for t in all_tools if t.name in ["web_search", "fetch_webpage", "wikipedia_search"]]
    document_tools = [t for t in all_tools if t.name in ["write_document", "read_document", "append_to_document", "list_documents"]]
    
    print(f"\n📦 Loaded Tools:")
    print(f"   🔍 Research tools: {[t.name for t in research_tools]}")
    print(f"   📄 Document tools: {[t.name for t in document_tools]}")
    
    # Create specialized agents
    researcher = ResearcherAgent(tools=research_tools)
    writer = WriterAgent(tools=document_tools)
    reviewer = ReviewerAgent(tools=document_tools)
    
    # Build the graph
    builder = StateGraph(AgentState)
    
    # Add agent nodes
    builder.add_node("researcher", researcher.run)
    builder.add_node("writer", writer.run)
    builder.add_node("reviewer", reviewer.run)
    
    # Define the pipeline flow: START → researcher → writer → reviewer → END
    builder.add_edge(START, "researcher")
    builder.add_edge("researcher", "writer")
    builder.add_edge("writer", "reviewer")
    builder.add_edge("reviewer", END)
    
    # Compile the graph
    graph = builder.compile()
    
    return graph


async def run_multi_agent(query: str):
    """
    Run the multi-agent pipeline with a given research query.
    
    Args:
        query: The research/writing task for the agents
    """
    print("\n" + "=" * 70)
    print("🤖 Multi-Agent Web Research and Document Writer")
    print("=" * 70)
    print(f"\n📝 Task: {query}\n")
    print("\n🔄 Pipeline: Researcher → Writer → Reviewer\n")
    
    # Create the multi-agent graph
    graph = await create_multi_agent_graph()
    
    # Initialize state
    initial_state = {
        "messages": [HumanMessage(content=query)],
        "research_data": "",
        "draft_document": "",
        "review_feedback": "",
        "final_document": "",
        "current_phase": "research"
    }
    
    # Run the pipeline
    result = await graph.ainvoke(initial_state)
    
    # Print final results
    print("\n" + "=" * 70)
    print("📊 PIPELINE COMPLETE")
    print("=" * 70)
    print(f"\n✅ Current Phase: {result.get('current_phase', 'unknown')}")
    print(f"\n📝 Review Summary:\n{result.get('review_feedback', 'No feedback')[:500]}...")
    
    return result


async def main():
    """Main entry point for the multi-agent system."""
    
    # Check for API key
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ Error: OPENAI_API_KEY environment variable not set")
        print("   Please set it in .env file or as an environment variable")
        return
    
    # Example research task
    query = """
    Research the current state of artificial general intelligence (AGI) development.
    Include:
    - Major companies and research labs working on AGI
    - Recent breakthroughs and milestones
    - Challenges and timelines predicted by experts
    
    Write a comprehensive summary document called 'agi_research_summary.md'
    """
    
    await run_multi_agent(query)


if __name__ == "__main__":
    asyncio.run(main())
