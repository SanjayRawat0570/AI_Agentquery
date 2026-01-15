# How to Run the Project

## Prerequisites
- Python 3.10+ (3.12.2 recommended)
- Virtual environment (optional but recommended)

## Quick Start (Recommended - Lightweight Mode)

### 1. Navigate to Project Directory
```powershell
cd 'c:\Users\ftt\Desktop\zango_aiagent\zango_aiagent\Project L2'
```

### 2. Create and Activate Virtual Environment (if not already done)
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

### 3. Install Lightweight Dependencies
```powershell
pip install -r requirements-lite.txt
```

This installs:
- fastapi==0.104.1
- uvicorn==0.24.0
- python-dotenv==1.0.0
- requests==2.31.0
- markdown==3.5.1
- pydantic==1.10.24
- pytest==7.4.0

### 4. Start Mock LLM Server (Terminal 1)
```powershell
python -m uvicorn mock_llm:app --host 127.0.0.1 --port 11435
```

**Output should show:**
```
INFO:     Started server process [XXXX]
INFO:     Uvicorn running on http://127.0.0.1:11435 (Press CTRL+C to quit)
```

### 5. Start Main Orchestrator Server (Terminal 2)
```powershell
python -m uvicorn main:app --host 127.0.0.1 --port 8000
```

**Output should show:**
```
INFO:     Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)
```

---

## Access the Application

Once both servers are running, you can access:

### Web UI (Recommended)
- **Interactive Test UI:** http://127.0.0.1:8000/static/ui.html
- **Minimal UI:** http://127.0.0.1:8000/ui

### API Documentation
- **Swagger UI (Interactive Docs):** http://127.0.0.1:8000/docs
- **ReDoc (Alternative Docs):** http://127.0.0.1:8000/redoc

### API Endpoints
- **POST /api/query** - Send a customer inquiry
- **GET /api/accounts/{account_id}** - Get account info
- **POST /api/accounts/{account_id}/change_subscription** - Change subscription plan

---

## Full Installation (Optional - With ML Features)

If you need chromadb and vector databases, install full dependencies:

```powershell
pip install -r requirements.txt
```

This includes:
- chromadb==0.4.17
- sentence-transformers==2.2.2
- langchain==0.1.3
- And other ML/vector DB packages

---

## Example API Call

### Using curl or Postman:
```bash
curl -X POST http://127.0.0.1:8000/api/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Can you compare CM-Pro and CM-Enterprise pricing?",
    "conversation_id": "conv-1234"
  }'
```

### Expected Response:
```json
{
  "response": "CM-Pro offers... CM-Enterprise provides...",
  "agent": "product",
  "conversation_id": "conv-1234"
}
```

---

## Troubleshooting

### Port Already in Use
If port 8000 or 11435 is already in use, use a different port:
```powershell
python -m uvicorn main:app --host 127.0.0.1 --port 8001
python -m uvicorn mock_llm:app --host 127.0.0.1 --port 11436
```

### chromadb Import Error
This is expected in lightweight mode. The system falls back to file-based knowledge base. If you need vector DB, install full requirements.

### Module Not Found
Make sure your virtual environment is activated:
```powershell
.venv\Scripts\Activate.ps1
```

---

## Project Structure

```
Project L2/
├── main.py                    # Main FastAPI orchestrator
├── mock_llm.py               # Mock LLM server
├── agent_implementations.py   # Agent logic
├── data_utils.py             # Data management
├── requirements.txt          # Full dependencies (with ML)
├── requirements-lite.txt     # Lightweight dependencies
├── static/
│   └── ui.html               # Web test UI
├── data/
│   ├── customer_conversations.jsonl
│   ├── faq.json
│   ├── product_catalog.json
│   └── tech_documentation.md
└── chroma_db/                # Vector DB storage (if installed)
```

---

## Keep Both Servers Running

**Important:** Both servers must run simultaneously:
- **Mock LLM Server** (port 11435): Provides LLM responses
- **Main Server** (port 8000): Handles API requests and orchestrates agents

Keep both terminal windows open while using the application.
