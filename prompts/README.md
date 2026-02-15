# Prompts Directory

This directory contains all agent system prompts as markdown files.

## Structure

- `researcher.md` - System prompt for the Researcher Agent
- `writer.md` - System prompt for the Writer Agent
- `reviewer.md` - System prompt for the Reviewer Agent
- `__init__.py` - Utility module with `load_prompt()` function

## Usage

Agents load their prompts using the `load_prompt()` function:

```python
from prompts import load_prompt

# In agent class
@property
def system_prompt(self) -> str:
    return load_prompt("researcher")  # Loads researcher.md
```

## Benefits

- **Easy Editing**: Modify prompts without touching Python code
- **Version Control**: Track prompt changes separately from code
- **Reusability**: Share prompts across different agents if needed
- **Clarity**: Prompts are easier to read and edit in markdown format
