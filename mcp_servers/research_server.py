"""
MCP Server for Web Research Tools

Provides tools for web searching, URL fetching, and Wikipedia queries.
Uses FastMCP for easy MCP server creation.
"""

from mcp.server.fastmcp import FastMCP
from duckduckgo_search import DDGS
import wikipedia
import requests
from bs4 import BeautifulSoup
from typing import Optional

# Initialize FastMCP server
mcp = FastMCP("Research")


@mcp.tool()
def web_search(query: str, max_results: int = 5) -> str:
    """
    Search the web using DuckDuckGo and return relevant results.
    
    Args:
        query: The search query string
        max_results: Maximum number of results to return (default: 5)
    
    Returns:
        Formatted search results with titles, snippets, and URLs
    """
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results))
        
        if not results:
            return f"No results found for query: {query}"
        
        formatted_results = []
        for i, result in enumerate(results, 1):
            formatted_results.append(
                f"{i}. **{result.get('title', 'No title')}**\n"
                f"   URL: {result.get('href', 'No URL')}\n"
                f"   {result.get('body', 'No description')}\n"
            )
        
        return f"Search results for '{query}':\n\n" + "\n".join(formatted_results)
    
    except Exception as e:
        return f"Error performing web search: {str(e)}"


@mcp.tool()
def fetch_webpage(url: str, max_chars: int = 5000) -> str:
    """
    Fetch and extract the main text content from a webpage URL.
    
    Args:
        url: The URL to fetch content from
        max_chars: Maximum characters to return (default: 5000)
    
    Returns:
        Extracted text content from the webpage
    """
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Remove script and style elements
        for element in soup(['script', 'style', 'nav', 'footer', 'header']):
            element.decompose()
        
        # Get text content
        text = soup.get_text(separator='\n', strip=True)
        
        # Clean up whitespace
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        text = '\n'.join(lines)
        
        # Truncate if needed
        if len(text) > max_chars:
            text = text[:max_chars] + "\n\n[Content truncated...]"
        
        return f"Content from {url}:\n\n{text}"
    
    except requests.exceptions.Timeout:
        return f"Error: Request timed out for URL: {url}"
    except requests.exceptions.RequestException as e:
        return f"Error fetching webpage: {str(e)}"
    except Exception as e:
        return f"Error processing webpage: {str(e)}"


@mcp.tool()
def wikipedia_search(query: str, sentences: int = 5) -> str:
    """
    Search Wikipedia and return a summary of the topic.
    
    Args:
        query: The topic to search for on Wikipedia
        sentences: Number of sentences to return in summary (default: 5)
    
    Returns:
        Wikipedia summary and related information
    """
    try:
        # Search for matching pages
        search_results = wikipedia.search(query, results=3)
        
        if not search_results:
            return f"No Wikipedia articles found for: {query}"
        
        # Try to get the summary for the first result
        try:
            page = wikipedia.page(search_results[0])
            summary = wikipedia.summary(search_results[0], sentences=sentences)
            
            return (
                f"**{page.title}**\n\n"
                f"{summary}\n\n"
                f"URL: {page.url}\n\n"
                f"Related topics: {', '.join(search_results[1:]) if len(search_results) > 1 else 'None'}"
            )
        
        except wikipedia.DisambiguationError as e:
            # Handle disambiguation pages
            options = e.options[:5]
            return (
                f"Multiple Wikipedia articles found for '{query}':\n"
                f"- " + "\n- ".join(options) + "\n\n"
                f"Please be more specific in your query."
            )
    
    except Exception as e:
        return f"Error searching Wikipedia: {str(e)}"


if __name__ == "__main__":
    # Run the MCP server using stdio transport
    mcp.run(transport="stdio")
