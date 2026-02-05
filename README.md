# Multi-Agent Web Research & Document Writer

A LangGraph-based multi-agent system that researches topics and creates comprehensive documents.

## Architecture

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│ Researcher  │ →  │   Writer    │ →  │  Reviewer   │
│   Agent     │    │   Agent     │    │   Agent     │
└─────────────┘    └─────────────┘    └─────────────┘
      ↓                  ↓                  ↓
 Web Search        Synthesize         Fact-check
 Wikipedia         into doc           & finalize
 URL Fetch
```

## Setup

### 1. Prerequisites
- Python 3.12+
- OpenAI API key

### 2. Create Virtual Environment (WSL Ubuntu)

```bash
# Navigate to project directory
cd '/mnt/c/Users/rsamarth/OneDrive - ZeOmega/Desktop/project_demo'

# Set Python 3.12 with pyenv
pyenv local 3.12

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Configure Environment

```bash
# Copy example env file
cp .env.example .env

# Edit .env and add your OpenAI API key
# OPENAI_API_KEY=sk-your-key-here
```

## Usage

### Run the Agent

```bash
source venv/bin/activate
python main.py
```

### Custom Research Query

Edit the `query` variable in `main.py`:

```python
query = """
Research [YOUR TOPIC HERE].
Include:
- Key points
- Recent developments
- Expert opinions

Write a summary document called 'my_research.md'
"""
```

## Project Structure

```
project_demo/
├── agents/                 # Multi-agent module
│   ├── __init__.py
│   ├── state.py           # Shared agent state
│   ├── base.py            # Abstract base agent
│   ├── researcher.py      # Gathers information
│   ├── writer.py          # Creates documents
│   └── reviewer.py        # Fact-checks & edits
├── mcp_servers/           # MCP tool servers
│   ├── research_server.py # Web search, Wikipedia, URL fetch
│   └── document_server.py # File read/write operations
├── output/                # Generated documents
├── main.py                # Entry point & orchestration
├── tools.py               # MCP client configuration
├── requirements.txt       # Python dependencies
└── .env                   # API keys (create from .env.example)
```

## Available Tools

### Research Tools
| Tool | Description |
|------|-------------|
| `web_search` | Search the web via DuckDuckGo |
| `fetch_webpage` | Extract content from URLs |
| `wikipedia_search` | Query Wikipedia for summaries |

### Document Tools
| Tool | Description |
|------|-------------|
| `write_document` | Create/overwrite a document |
| `read_document` | Read document contents |
| `append_to_document` | Append to existing document |
| `list_documents` | List all documents |
| `delete_document` | Delete a document |

## Output

Generated documents are saved to the `output/` directory.
