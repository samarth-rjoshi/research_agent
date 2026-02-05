"""
MCP Server for Document Writing Tools

Provides tools for creating, reading, and managing documents.
Uses FastMCP for easy MCP server creation.
"""

from mcp.server.fastmcp import FastMCP
from pathlib import Path
from datetime import datetime
from typing import List
import os

# Initialize FastMCP server
mcp = FastMCP("Document")

# Default output directory (relative to project root)
OUTPUT_DIR = Path(__file__).parent.parent / "output"


def ensure_output_dir():
    """Ensure the output directory exists."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


@mcp.tool()
def write_document(filename: str, content: str) -> str:
    """
    Write content to a document file. Creates a new file or overwrites existing.
    
    Args:
        filename: Name of the document file (e.g., 'research_notes.md')
        content: The content to write to the document
    
    Returns:
        Confirmation message with file path
    """
    try:
        ensure_output_dir()
        
        # Sanitize filename
        safe_filename = "".join(c for c in filename if c.isalnum() or c in '._- ')
        if not safe_filename:
            safe_filename = f"document_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
        
        filepath = OUTPUT_DIR / safe_filename
        
        # Write content
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        
        return f"✅ Document written successfully!\n\nFile: {filepath}\nSize: {len(content)} characters"
    
    except Exception as e:
        return f"❌ Error writing document: {str(e)}"


@mcp.tool()
def read_document(filename: str) -> str:
    """
    Read the content of an existing document.
    
    Args:
        filename: Name of the document file to read
    
    Returns:
        The content of the document or an error message
    """
    try:
        filepath = OUTPUT_DIR / filename
        
        if not filepath.exists():
            return f"❌ Document not found: {filename}\n\nAvailable documents:\n{list_documents()}"
        
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        return f"📄 Content of '{filename}':\n\n{content}"
    
    except Exception as e:
        return f"❌ Error reading document: {str(e)}"


@mcp.tool()
def append_to_document(filename: str, content: str) -> str:
    """
    Append content to an existing document. Creates the file if it doesn't exist.
    
    Args:
        filename: Name of the document file
        content: The content to append
    
    Returns:
        Confirmation message
    """
    try:
        ensure_output_dir()
        
        filepath = OUTPUT_DIR / filename
        
        # Check if file exists
        existed = filepath.exists()
        
        # Append content
        with open(filepath, 'a', encoding='utf-8') as f:
            if existed:
                f.write("\n\n")  # Add spacing if appending
            f.write(content)
        
        action = "appended to" if existed else "created with"
        return f"✅ Content {action} document: {filename}"
    
    except Exception as e:
        return f"❌ Error appending to document: {str(e)}"


@mcp.tool()
def list_documents() -> str:
    """
    List all documents in the output directory.
    
    Returns:
        Formatted list of available documents with their sizes and dates
    """
    try:
        ensure_output_dir()
        
        files = list(OUTPUT_DIR.iterdir())
        
        if not files:
            return "📂 No documents found in output directory."
        
        doc_list = []
        for f in sorted(files):
            if f.is_file():
                stat = f.stat()
                size = stat.st_size
                modified = datetime.fromtimestamp(stat.st_mtime).strftime('%Y-%m-%d %H:%M')
                doc_list.append(f"  • {f.name} ({size} bytes, modified: {modified})")
        
        if not doc_list:
            return "📂 No documents found in output directory."
        
        return f"📂 Available documents:\n\n" + "\n".join(doc_list)
    
    except Exception as e:
        return f"❌ Error listing documents: {str(e)}"


@mcp.tool()
def delete_document(filename: str) -> str:
    """
    Delete a document from the output directory.
    
    Args:
        filename: Name of the document to delete
    
    Returns:
        Confirmation or error message
    """
    try:
        filepath = OUTPUT_DIR / filename
        
        if not filepath.exists():
            return f"❌ Document not found: {filename}"
        
        filepath.unlink()
        return f"🗑️ Document deleted: {filename}"
    
    except Exception as e:
        return f"❌ Error deleting document: {str(e)}"


if __name__ == "__main__":
    # Run the MCP server using stdio transport
    mcp.run(transport="stdio")
