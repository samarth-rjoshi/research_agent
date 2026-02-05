"""
MCP Client Configuration for Tools

Sets up MultiServerMCPClient to connect to research and document MCP servers.
Provides async access to all MCP tools for the LangGraph agent.
"""

from langchain_mcp_adapters.client import MultiServerMCPClient
from pathlib import Path
import os

# Get absolute paths to MCP servers
PROJECT_DIR = Path(__file__).parent
RESEARCH_SERVER = str(PROJECT_DIR / "mcp_servers" / "research_server.py")
DOCUMENT_SERVER = str(PROJECT_DIR / "mcp_servers" / "document_server.py")


def get_mcp_client() -> MultiServerMCPClient:
    """
    Create and return a MultiServerMCPClient configured with all MCP servers.
    
    Returns:
        MultiServerMCPClient instance connected to research and document servers
    """
    client = MultiServerMCPClient(
        {
            "research": {
                "command": "python",
                "args": [RESEARCH_SERVER],
                "transport": "stdio",
            },
            "document": {
                "command": "python",
                "args": [DOCUMENT_SERVER],
                "transport": "stdio",
            }
        }
    )
    return client


async def get_tools():
    """
    Async function to get all tools from MCP servers.
    
    Returns:
        List of LangChain tools from all connected MCP servers
    """
    client = get_mcp_client()
    tools = await client.get_tools()
    return tools
