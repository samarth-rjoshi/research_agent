"""
PDF Generation Utility

Provides functions to write plain text content to PDF documents using fpdf.
"""

from pathlib import Path
from datetime import datetime
from fpdf import FPDF
import os

# Default output directory (relative to project root)
PROJECT_DIR = Path(__file__).parent.parent
OUTPUT_DIR = PROJECT_DIR / "output"


def ensure_output_dir():
    """Ensure the output directory exists."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def write_pdf(filename: str, content: str) -> str:
    """
    Write plain text content to a PDF document.

    Args:
        filename: Name of the document file (e.g., 'report.pdf'). A .pdf extension will be added if missing.
        content: The plain text content to write to the document.

    Returns:
        Confirmation message with file path.
    """
    try:
        ensure_output_dir()

        # Ensure .pdf extension
        safe_filename = "".join(c for c in filename if c.isalnum() or c in '._- ')
        if not safe_filename:
            safe_filename = f"document_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        if not safe_filename.lower().endswith(".pdf"):
            safe_filename = safe_filename.rsplit(".", 1)[0] + ".pdf"

        filepath = OUTPUT_DIR / safe_filename

        # Build a simple PDF
        pdf = FPDF()
        pdf.set_auto_page_break(auto=True, margin=20)
        pdf.add_page()
        pdf.set_font("Helvetica", size=11)

        # Replace common Unicode characters with ASCII equivalents
        sanitized = content
        replacements = {
            "\u2013": "-",   # en dash
            "\u2014": "--",  # em dash
            "\u2018": "'",   # left single quote
            "\u2019": "'",   # right single quote
            "\u201c": '"',   # left double quote
            "\u201d": '"',   # right double quote
            "\u2026": "...", # ellipsis
            "\u2022": "*",   # bullet
            "\u00a0": " ",   # non-breaking space
        }
        for uni_char, ascii_char in replacements.items():
            sanitized = sanitized.replace(uni_char, ascii_char)
        
        # Strip any remaining non-latin1 characters
        sanitized = sanitized.encode("latin-1", errors="replace").decode("latin-1")

        pdf.multi_cell(0, 6, sanitized)
        pdf.output(str(filepath))

        return f"PDF written successfully! File: {filepath}, Size: {filepath.stat().st_size} bytes"

    except Exception as e:
        return f"Error writing document: {str(e)}"
