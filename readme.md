# Enterprise Customer Support Agent Orchestrator

A personal project implementing a multi-agent AI system for intelligent customer support automation. This system demonstrates an orchestration architecture that efficiently handles diverse customer inquiries through specialized AI agents.

## Project Overview

This is an intelligent multi-agent orchestrator that routes and processes customer support queries using specialized agents:
- **Router Agent**: Classifies and directs inquiries to the appropriate specialist
- **Product Specialist**: Handles product information, pricing, and feature comparisons
- **Technical Support**: Provides troubleshooting and technical assistance
- **Order/Billing Agent**: Manages order status and billing inquiries
- **Account Management**: Handles subscription changes and account updates

The system includes a lightweight FastAPI server with both REST API and browser-based UI for testing.

## Features

- Multi-agent orchestration with intelligent query routing
- Browser-based test UI for interactive testing
- Mock LLM server for offline/demo responses
- File-based knowledge base with optional vector DB support
- CORS-enabled REST API
- In-memory account management for subscription demos
- Graceful fallbacks when external services are unavailable

---

## Quick Start

### Prerequisites
- Python 3.10+ recommended
- Virtual environment (recommended)

### Installation Options

There are two dependency sets:
- **requirements.txt** — Full stack with vector DB support (chromadb, sentence-transformers, numpy). May require native build tools.
- **requirements-lite.txt** — Lightweight dependencies for quick testing without heavy ML packages (recommended for demo).

### Setup (Fast Demo Mode)

1. **Create and activate virtual environment** (PowerShell):
```powershell
cd 'C:\Users\ftt\Desktop\AI-agent'
python -m venv .venv
.venv\Scripts\Activate.ps1
```

2. **Install dependencies**:
```powershell
pip install -r requirements-lite.txt
```

3. **Start the mock LLM server** (provides `/api/generate` endpoint):
```powershell
python -m uvicorn mock_llm:app --host 127.0.0.1 --port 11434
```
Leave this running in a separate terminal.

4. **Start the main orchestrator server**:
```powershell
python -m uvicorn main:app --host 127.0.0.1 --port 8000
```

5. **Access the test UI**:
- http://127.0.0.1:8000/static/ui.html (full static test page)
- http://127.0.0.1:8000/ui (minimal built-in UI)
- Or open `ui_test.html` from the project root

---

## API Endpoints

### Query Endpoint
**POST /api/query**
```json
{
  "query": "Can you compare CM-Pro and CM-Enterprise pricing?",
  "conversation_id": "conv-1234"  // optional
}
```
Response:
```json
{
  "response": "...",
  "agent": "product",
  "conversation_id": "conv-1234"
}
```

### Account Management
**POST /api/accounts/{account_id}/change_subscription**
```json
{ "plan": "cm-enterprise" }
```

**GET /api/accounts/{account_id}**
- Returns demo accounts (ACC-1111, ACC-2222)

### Technical Support
**POST /api/diagnose**
```json
{ "description": "Seeing Error E1234 when calling the API" }
```

### Order Management
**GET /api/orders/{order_id}**
- Mock order data for ORD-12345 and ORD-56789

### Admin
**POST /api/admin/add_account_agent**
- Dynamically enable Account Management Agent

---

## Architecture & Implementation

### Components

**main.py**
- FastAPI server with CORS middleware for browser access
- Static file serving and UI endpoints (`/static/ui.html` and `/ui`)
- In-memory `ACCOUNTS` store for demo subscription changes
- Falls back to file-based knowledge base when `chromadb` is not installed

**agent_implementations.py**
- Asynchronous agent implementations with robust, rule-based router
- `LLMUtils` with automatic availability detection and graceful fallbacks
- Product, Technical, Billing, and Account agents with file-KB fallbacks when vector DB/LLM unavailable

**mock_llm.py**
- Lightweight FastAPI app implementing `/api/generate` endpoint
- Simulates LLM responses for local testing without external dependencies

**static/ui.html**
- Browser-based test interface for exercising all endpoints
- Interactive testing with JSON response display

**ui_test.html**
- Standalone test page for quick manual testing
- Uses absolute URLs for flexible deployment

**data/**
- Product catalog, FAQs, and technical documentation
- Knowledge base files for agent responses

---

## Fallback Behavior & Troubleshooting

### Vector DB Fallback
If `chromadb` or vector DB isn't installed, the server loads the product catalog, FAQs, and tech docs from `data/` directory and uses keyword matching for queries.

### LLM Fallback
When the LLM endpoint configured by `OLLAMA_BASE_URL` is unreachable, `LLMUtils` marks it as unavailable and agents return friendly fallback messages while attempting to use local knowledge.

### Windows Installation Issues
Common issues when installing full dependencies:
- `numpy` and `sentence-transformers` may require C build toolchain
- **Solutions:**
  - Use conda: `conda install -c conda-forge numpy sentence-transformers`
  - Use Docker with a Linux image
  - Stick with `requirements-lite.txt` for demo mode

---

## Testing

### Manual Testing
Use the browser UI at http://127.0.0.1:8000/static/ui.html or `ui_test.html` to exercise all endpoints.

### CLI Testing (PowerShell)
```powershell
# Query the agent
$body = @{ query = 'What is the monthly price for CM-Pro and what features does it include?'; conversation_id = 'conv-1001' } | ConvertTo-Json
Invoke-RestMethod -Uri http://127.0.0.1:8000/api/query -Method Post -Body $body -ContentType 'application/json'

# Change subscription
$body = @{ plan = 'cm-enterprise' } | ConvertTo-Json
Invoke-RestMethod -Uri http://127.0.0.1:8000/api/accounts/ACC-1111/change_subscription -Method Post -Body $body -ContentType 'application/json'
```

---

## Future Enhancements

- [ ] Periodic LLM health checks with automatic reconnection
- [ ] Admin endpoint for service status monitoring
- [ ] Chat history and conversation timeline in UI
- [ ] Streaming response support
- [ ] Persistent vector DB integration (chromadb in Docker/conda environment)
- [ ] Enhanced RAG flows with better context retrieval
- [ ] User authentication and session management

---

## Project Structure

```
AI-agent/
├── agent_implementations.py     # Core agent logic
├── main.py                      # FastAPI server
├── mock_llm.py                  # Mock LLM service
├── data_utils.py                # Data handling utilities
├── requirements.txt             # Full dependencies
├── requirements-lite.txt        # Lightweight dependencies
├── Dockerfile                   # Container configuration
├── setup.py                     # Package setup
├── data/                        # Knowledge base
│   ├── product_catalog.json
│   ├── faq.json
│   ├── tech_documentation.md
│   └── customer_conversations.jsonl
├── static/
│   └── ui.html                  # Test interface
├── tests/
│   ├── automated_tester.py
│   └── test_scenarios.md
├── chroma_db/                   # Vector DB storage (when enabled)
└── ui_test.html                 # Quick test page
```

---

## License

This is a personal project for demonstration purposes.

## Contact

For questions or collaboration opportunities, feel free to reach out!
