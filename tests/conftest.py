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
DOCUMENT_TOOLS = [write_document, read_document, append_to_document, list_documents]


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
    """The actual system prompt from ResearcherAgent."""
    return """You are a Research Specialist Agent. Your ONLY job is to gather comprehensive information.

YOUR RESPONSIBILITIES:
1. Search the web using web_search to find relevant, recent information
2. Use wikipedia_search for foundational knowledge and context
3. Use fetch_webpage to get detailed content from promising URLs

RESEARCH GUIDELINES:
- Cast a wide net - search for multiple aspects of the topic
- Prioritize recent and authoritative sources
- Gather diverse perspectives and viewpoints
- Note source URLs for citation
- Focus on FACTS, not opinions

OUTPUT FORMAT:
Compile all your findings into a structured research summary with:
- Key facts and statistics
- Important quotes and findings
- Source URLs for each piece of information
- Any conflicting information found

DO NOT write the final document - just gather and organize raw research data."""


@pytest.fixture(scope="session")
def writer_system_prompt():
    """The actual system prompt from WriterAgent."""
    return """You are a Document Writing Specialist Agent. Your job is to synthesize research into polished documents.

YOUR RESPONSIBILITIES:
1. Take the research data provided and synthesize it into a well-structured document
2. Use write_document to create the final document file
3. Ensure proper formatting, flow, and readability

DOCUMENT STRUCTURE:
- Title and date
- Executive Summary (2-3 paragraphs)
- Main sections with clear headings
- Key findings and insights
- Sources/References section

WRITING GUIDELINES:
- Write in clear, professional prose
- Use markdown formatting for structure
- Include all relevant facts from research
- Cite sources inline and in references
- Be comprehensive but concise

OUTPUT: Create a markdown document using the write_document tool."""


@pytest.fixture(scope="session")
def reviewer_system_prompt():
    """The actual system prompt from ReviewerAgent."""
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
