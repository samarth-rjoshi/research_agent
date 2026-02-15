"""
Shared fixtures for live LangSmith pytest test suite.

NO MOCKING — uses real ChatOpenAI and real tool schemas.
Tests make actual LLM API calls to verify tool selection and argument generation.

Requires OPENAI_API_KEY to be set.
"""

import os
import sys
import pytest
from typing import List, Optional

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage

# Ensure project root is on path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Load .env from project root
load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))


# ---------------------------------------------------------------------------
# Real tool definitions (matching MCP server signatures exactly)
# These give the LLM the correct schemas for tool binding.
# ---------------------------------------------------------------------------

@tool
def web_search(query: str, max_results: int = 5) -> str:
    """Search the web using DuckDuckGo and return relevant results.

    Args:
        query: The search query string
        max_results: Maximum number of results to return (default: 5)
    """
    return f"Search results for '{query}'"


@tool
def fetch_webpage(url: str, max_chars: int = 5000) -> str:
    """Fetch and extract the main text content from a webpage URL.

    Args:
        url: The URL to fetch content from
        max_chars: Maximum characters to return (default: 5000)
    """
    return f"Content from {url}"


@tool
def wikipedia_search(query: str, sentences: int = 5) -> str:
    """Search Wikipedia and return a summary of the topic.

    Args:
        query: The topic to search for on Wikipedia
        sentences: Number of sentences to return in summary (default: 5)
    """
    return f"Wikipedia summary for '{query}'"


@tool
def write_document(filename: str, content: str) -> str:
    """Write content to a document file. Creates a new file or overwrites existing.

    Args:
        filename: Name of the document file (e.g., 'research_notes.md')
        content: The content to write to the document
    """
    return f"Document '{filename}' written successfully"


@tool
def read_document(filename: str) -> str:
    """Read the content of an existing document.

    Args:
        filename: Name of the document file to read
    """
    return f"Content of '{filename}'"


@tool
def append_to_document(filename: str, content: str) -> str:
    """Append content to an existing document. Creates the file if it doesn't exist.

    Args:
        filename: Name of the document file
        content: The content to append
    """
    return f"Content appended to '{filename}'"


@tool
def list_documents() -> str:
    """List all documents in the output directory."""
    return "No documents found."


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

RESEARCH_TOOLS = [web_search, fetch_webpage, wikipedia_search]
DOCUMENT_TOOLS = [write_document]


@pytest.fixture(scope="session")
def llm():
    """Real ChatOpenAI instance (session-scoped to reuse connections)."""
    return ChatOpenAI(model="gpt-4o", temperature=0.1)


@pytest.fixture(scope="session")
def research_model(llm):
    """LLM bound with research tool schemas."""
    return llm.bind_tools(RESEARCH_TOOLS)


@pytest.fixture(scope="session")
def document_model(llm):
    """LLM bound with document tool schemas."""
    return llm.bind_tools(DOCUMENT_TOOLS)


@pytest.fixture(scope="session")
def researcher_system_prompt():
    """Load the actual researcher prompt from file."""
    from prompts import load_prompt
    return load_prompt("researcher")


@pytest.fixture(scope="session")
def writer_system_prompt():
    """Load the actual writer prompt from file."""
    from prompts import load_prompt
    return load_prompt("writer")


@pytest.fixture(scope="session")
def reviewer_system_prompt():
    """Reviewer prompt for the legacy pipeline test (no separate file needed)."""
    return """You are a Document Review Specialist Agent. Your job is to fact-check and improve documents.

YOUR RESPONSIBILITIES:
1. Read the current document using read_document
2. Review for accuracy, completeness, and quality
3. Make improvements and save the final version using write_document

REVIEW CHECKLIST:
- Verify facts are accurately represented
- Check that sources are properly cited
- Ensure logical flow and structure
- Improve clarity and readability
- Fix any grammatical or formatting issues
- Add any missing important information

OUTPUT:
1. Provide a brief review summary of changes made
2. Save the improved document using write_document

Be thorough but efficient. Focus on substantive improvements."""

