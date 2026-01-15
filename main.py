import json
import logging
import os
from typing import Any, Dict, List, Optional

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from agent_implementations import LLMUtils
from agent_implementations import AgentOrchestrator
# Import DataManager lazily later to avoid hard dependency on chromadb/langchain at import time

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(title="TechSolutions Support Agent Orchestrator")

# Enable CORS so browser-based UIs (file served or different origin) can call the API.
# Allowing all origins here for convenience in testing/demo; tighten in production.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve a static folder so we can open an HTML test page from the same origin
app.mount("/static", StaticFiles(directory="static"), name="static")

# Initialize Data Manager lazily so the server can run in demo/offline mode when chromadb/langchain
# are not installed. If DataManager fails to import or initialize, fall back to loading local files
# and set vector_db collections to None so agents use file-based fallbacks.
knowledge_base = None
vector_db = None
try:
    # Try to import and initialize DataManager (requires chromadb/langchain)
    from data_utils import DataManager

    data_manager = DataManager(
        data_dir=os.getenv("DATA_DIR", "data"),
        db_dir=os.getenv("DB_DIR", "chroma_db"),
    )
    logger.info("DataManager initialized successfully")

    # Load knowledge base and prepare vector DB
    knowledge_base = data_manager.load_knowledge_base()
    vector_db = data_manager.prepare_vector_db(knowledge_base)
    logger.info("Vector databases initialized with collections")
except Exception as e:
    logger.warning(f"DataManager unavailable or failed to initialize: {e}")
    # Fallback: load KB files from data directory without vector DB
    data_dir = os.getenv("DATA_DIR", "data")
    try:
        # Load product catalog
        with open(os.path.join(data_dir, "product_catalog.json"), "r", encoding="utf-8") as f:
            product_catalog = json.load(f)
    except Exception:
        product_catalog = {}

    try:
        with open(os.path.join(data_dir, "faq.json"), "r", encoding="utf-8") as f:
            faqs = json.load(f)
    except Exception:
        faqs = {}

    tech_docs = ""
    try:
        with open(os.path.join(data_dir, "tech_documentation.md"), "r", encoding="utf-8") as f:
            tech_docs = f.read()
    except Exception:
        tech_docs = ""

    knowledge_base = {
        "product_catalog": product_catalog,
        "faqs": faqs,
        "tech_docs": tech_docs,
        "customer_conversations": [],
    }
    vector_db = {"products": None, "technical": None, "conversations": None}
    logger.info("Loaded local knowledge base files as fallback (vector_db=None)")


# In-memory accounts store used by mock endpoints. This allows subscription changes
# to persist during the process lifetime for demo purposes.
ACCOUNTS: Dict[str, Any] = {
    "ACC-1111": {
        "account_id": "ACC-1111",
        "name": "Acme Corp",
        "subscription": {
            "plan": "cm-pro",
            "status": "active",
            "start_date": "2023-01-15",
            "renewal_date": "2024-01-15",
            "payment_method": "credit_card",
            "auto_renew": True,
        },
        "users": [
            {"email": "admin@acme.example.com", "role": "admin"},
            {"email": "user1@acme.example.com", "role": "viewer"},
            {"email": "user2@acme.example.com", "role": "operator"},
        ],
    },
    "ACC-2222": {
        "account_id": "ACC-2222",
        "name": "Globex Inc",
        "subscription": {
            "plan": "cm-enterprise",
            "status": "active",
            "start_date": "2023-03-10",
            "renewal_date": "2024-03-10",
            "payment_method": "invoice",
            "auto_renew": False,
        },
        "users": [
            {"email": "admin@globex.example.com", "role": "admin"},
            {"email": "finance@globex.example.com", "role": "billing"},
            {"email": "security@globex.example.com", "role": "security_admin"},
            {"email": "devops@globex.example.com", "role": "operator"},
        ],
    },
}



OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
MODEL_NAME = os.getenv("MODEL_NAME", "gemma3:1b")


# Initialize LLM utils
try:
    llm = LLMUtils(
        base_url=OLLAMA_BASE_URL,
        model_name=MODEL_NAME,
    )
    logger.info("LLM initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize LLM: {e}")
    llm = None

# Initialize Agent Orchestrator
if llm and knowledge_base and vector_db:
    agent_orchestrator = AgentOrchestrator(
        llm_utils=llm,
        knowledge_base=knowledge_base,
        vector_db=vector_db
    )
    logger.info("Agent Orchestrator initialized successfully")
else:
    agent_orchestrator = None
    logger.error("Failed to initialize Agent Orchestrator due to missing components")


# Define API models
class CustomerQuery(BaseModel):
    query: str
    conversation_id: Optional[str] = None


class AgentResponse(BaseModel):
    response: str
    agent: str
    conversation_id: str


@app.post("/api/query", response_model=AgentResponse)
async def process_customer_query(query: CustomerQuery):
    """Process a customer support query"""
    try:
        if not agent_orchestrator:
            raise HTTPException(status_code=500, detail="System not initialized properly")
            
        result = await agent_orchestrator.process_query(query.query, query.conversation_id)
        return result
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/admin/add_account_agent")
async def add_account_agent():
    """Admin endpoint to add the Account Management Agent mid-session."""
    try:
        if not agent_orchestrator:
            raise HTTPException(status_code=500, detail="System not initialized properly")

        result = agent_orchestrator.add_account_management_agent()
        return {"status": "ok", "result": result}
    except Exception as e:
        logger.error(f"Error adding account management agent: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/admin/llm_status")
async def llm_status():
    """Return current LLM availability status."""
    try:
        if llm is None:
            return {"available": False, "detail": "LLM not configured"}
        return {"available": getattr(llm, "available", False)}
    except Exception as e:
        logger.error(f"Error retrieving LLM status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/admin/llm_recheck")
async def llm_recheck():
    """Force a re-check of the LLM endpoint health without restarting the server."""
    try:
        if llm is None:
            raise HTTPException(status_code=500, detail="LLM not configured")
        ok = llm.check_availability()
        return {"rechecked": True, "available": ok}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error rechecking LLM availability: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Mock API endpoints for testing
@app.get("/api/orders/{order_id}")
async def get_order(order_id: str):
    """Mock Order API endpoint"""
    # Mock order data
    orders = {
        "ORD-12345": {
            "order_id": "ORD-12345",
            "status": "shipped",
            "items": [{"product_id": "cm-pro", "quantity": 1, "price": 149.99}],
            "total": 149.99,
            "order_date": "2023-09-10",
            "shipping_date": "2023-09-12",
            "delivery_date": "2023-09-15",
        },
        "ORD-56789": {
            "order_id": "ORD-56789",
            "status": "processing",
            "items": [
                {"product_id": "cm-enterprise", "quantity": 1, "price": 499.99},
                {"product_id": "addon-premium-support", "quantity": 1, "price": 299.99},
            ],
            "total": 799.98,
            "order_date": "2023-09-22",
            "shipping_date": None,
            "delivery_date": None,
        },
    }

    if order_id in orders:
        return orders[order_id]
    else:
        raise HTTPException(status_code=404, detail="Order not found")


@app.get("/api/accounts/{account_id}")
async def get_account(account_id: str):
    """Mock Account API endpoint"""
    # Returns account data from the in-memory ACCOUNTS store
    if account_id in ACCOUNTS:
        return ACCOUNTS[account_id]
    else:
        raise HTTPException(status_code=404, detail="Account not found")


def update_account_subscription(account_id: str, new_plan: str) -> Dict[str, Any]:
    """Helper to update the subscription plan for an account in the in-memory store."""
    if account_id not in ACCOUNTS:
        raise KeyError("Account not found")
    acct = ACCOUNTS[account_id]
    acct.setdefault("subscription", {})["plan"] = new_plan
    # Update other fields to indicate pending change for demo purposes
    acct["subscription"]["status"] = "pending_change"
    return acct


@app.post("/api/accounts/{account_id}/change_subscription")
async def change_subscription(account_id: str, request: Request):
    """Mock endpoint to change an account's subscription plan (demo only)."""
    try:
        data = await request.json()
        new_plan = data.get("plan")
        if not new_plan:
            raise HTTPException(status_code=400, detail="Missing 'plan' in request body")
        try:
            updated = update_account_subscription(account_id, new_plan)
            return {"status": "ok", "account": updated}
        except KeyError:
            raise HTTPException(status_code=404, detail="Account not found")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error changing subscription: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/diagnose")
async def diagnose_issue(request: Request):
    """Mock troubleshooting API endpoint"""
    try:
        data = await request.json()
        logger.info(f"Received diagnose request data: {data}")
        logger.info("********************************************")
        issue_description = data.get("description", "")

        # Simple keyword matching for demo purposes
        if "error e1234" in issue_description.lower():
            return {
                "issue_id": "E1234",
                "name": "API Connection Failure",
                "solutions": [
                    "Verify API credentials in Settings > Connections",
                    "Check if your firewall allows outbound connections to cloud provider APIs",
                    "Ensure cloud provider services are operational",
                ],
                "documentation_link": "docs.techsolutions.example.com/errors/e1234",
            }
        elif "error e5678" in issue_description.lower():
            return {
                "issue_id": "E5678",
                "name": "Container Image Verification Failed",
                "solutions": [
                    "Check image integrity and re-pull from registry",
                    "Verify signature configuration in Security > Image Signing",
                    "Review scan results in Security > Vulnerability Reports",
                ],
                "documentation_link": "docs.techsolutions.example.com/errors/e5678",
            }
        else:
            return {
                "issue_id": "unknown",
                "name": "Unrecognized Issue",
                "solutions": [
                    "Check application logs for specific error messages",
                    "Verify your configuration settings",
                    "Contact support with error details for assistance",
                ],
                "documentation_link": "docs.techsolutions.example.com/troubleshooting",
            }
    except Exception as e:
        logger.error(f"Error in diagnose endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Custom exception handler
@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=500,
        content={"detail": str(exc)},
    )


# Application startup event
@app.on_event("startup")
async def startup_event():
    logger.info("Starting up Support Agent Orchestrator")
    # Add any additional startup tasks here


# Application shutdown event
@app.on_event("shutdown")
async def shutdown_event():
    logger.info("Shutting down Support Agent Orchestrator")
    # Add any cleanup tasks here


# Add a simple health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "healthy"}


@app.get("/ui", response_class=HTMLResponse)
async def web_ui():
        """Simple web UI for live testing the orchestrator endpoints."""
        html = """
        <!doctype html>
        <html>
            <head>
                <meta charset="utf-8" />
                <title>TechSolutions Support Agent - Live Test UI</title>
                <style>
                    body { font-family: Arial, sans-serif; max-width: 900px; margin: 2rem auto; }
                    textarea { width: 100%; height: 100px; }
                    .response { white-space: pre-wrap; background: #f7f7f7; padding: 1rem; border-radius: 6px; }
                    .row { margin-bottom: 1rem; }
                </style>
            </head>
            <body>
                <h1>TechSolutions Support Agent — Live Test UI</h1>

                <div class="row">
                    <label for="query">Customer query</label><br />
                    <textarea id="query">What is the monthly price for CM-Pro and what features does it include?</textarea>
                </div>

                <div class="row">
                    <button id="send">Send Query</button>
                    <button id="health">Health Check</button>
                </div>

                <h3>Response</h3>
                <div id="result" class="response">(Responses will appear here)</div>

                <hr />
                <h2>Account management</h2>
                <div class="row">
                    <label for="account">Account ID</label>
                    <input id="account" value="ACC-1111" />
                </div>
                <div class="row">
                    <label for="plan">New plan</label>
                    <input id="plan" value="cm-enterprise" />
                </div>
                <div class="row">
                    <button id="change">Change subscription</button>
                    <button id="getacct">Get account</button>
                </div>
                <div id="acctresult" class="response">(Account responses)</div>

                <script>
                    async function postJson(url, body){
                        const r = await fetch(url, { method: 'POST', headers: {'Content-Type':'application/json'}, body: JSON.stringify(body)});
                        return r.json();
                    }

                    document.getElementById('send').onclick = async () => {
                        const q = document.getElementById('query').value;
                        try{
                            const res = await postJson('/api/query', { query: q });
                            document.getElementById('result').textContent = JSON.stringify(res, null, 2);
                        }catch(e){ document.getElementById('result').textContent = 'Error: '+e; }
                    }

                    document.getElementById('health').onclick = async () => {
                        try{
                            const r = await fetch('/health');
                            const j = await r.json();
                            document.getElementById('result').textContent = JSON.stringify(j, null, 2);
                        }catch(e){ document.getElementById('result').textContent = 'Error: '+e; }
                    }

                    document.getElementById('change').onclick = async () => {
                        const acct = document.getElementById('account').value;
                        const plan = document.getElementById('plan').value;
                        try{
                            const res = await postJson(`/api/accounts/${acct}/change_subscription`, { plan });
                            document.getElementById('acctresult').textContent = JSON.stringify(res, null, 2);
                        }catch(e){ document.getElementById('acctresult').textContent = 'Error: '+e; }
                    }

                    document.getElementById('getacct').onclick = async () => {
                        const acct = document.getElementById('account').value;
                        try{
                            const r = await fetch(`/api/accounts/${acct}`);
                            const j = await r.json();
                            document.getElementById('acctresult').textContent = JSON.stringify(j, null, 2);
                        }catch(e){ document.getElementById('acctresult').textContent = 'Error: '+e; }
                    }
                </script>
            </body>
        </html>
        """
        return HTMLResponse(content=html)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
