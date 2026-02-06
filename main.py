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
from agent_implementations import AgentOrchestrator, tool_registry
# Import DataManager lazily later to avoid hard dependency on chromadb/langchain at import time

# Configure logging FIRST
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# PHASE 1: Import new components
from memory_manager import conversation_manager, ConversationMemory
from monitoring import agent_metrics, performance_monitor, health_check, track_performance
from datetime import datetime
import time

# Try to import database components (optional dependency)
try:
    from models import get_db_manager
    db_available = True
except Exception as e:
    db_available = False
    logger.warning(f"Database components not available: {e}")

# PHASE 2: Import advanced components
try:
    from react_agent import ReActAgent
    react_available = True
except Exception as e:
    react_available = False
    logger.warning(f"ReAct components not available: {e}")

try:
    from tool_integrations import integration_factory
    integrations_available = True
except Exception as e:
    integrations_available = False
    logger.warning(f"Tool integrations not available: {e}")

# PHASE 3: Import enterprise components
try:
    from agent_collaboration import agent_collaboration_hub
    collaboration_available = True
except Exception as e:
    collaboration_available = False
    logger.warning(f"Agent collaboration not available: {e}")

try:
    from dynamic_agent_factory import dynamic_agent_factory
    dynamic_agents_available = True
except Exception as e:
    dynamic_agents_available = False
    logger.warning(f"Dynamic agent factory not available: {e}")

try:
    from advanced_analytics import advanced_analytics_engine
    advanced_analytics_available = True
except Exception as e:
    advanced_analytics_available = False
    logger.warning(f"Advanced analytics not available: {e}")

try:
    from realtime_streaming import realtime_streaming_engine, StreamEventType
    streaming_available = True
except Exception as e:
    streaming_available = False
    logger.warning(f"Real-time streaming not available: {e}")

try:
    from semantic_search import semantic_search_engine
    semantic_search_available = True
except Exception as e:
    semantic_search_available = False
    logger.warning(f"Semantic search not available: {e}")

try:
    from admin_analytics import admin_analytics
    analytics_available = True
except Exception as e:
    analytics_available = False
    logger.warning(f"Admin analytics not available: {e}")

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
    product_catalog_path = os.getenv(
        "PRODUCT_CATALOG_PATH", os.path.join(data_dir, "product_catalog.json")
    )
    faq_path = os.getenv("FAQ_PATH", os.path.join(data_dir, "faq.json"))
    tech_docs_path = os.getenv(
        "TECH_DOCS_PATH", os.path.join(data_dir, "tech_documentation.md")
    )
    conversations_path = os.getenv(
        "CONVERSATIONS_PATH", os.path.join(data_dir, "customer_conversations.jsonl")
    )
    try:
        # Load product catalog
        with open(product_catalog_path, "r", encoding="utf-8") as f:
            product_catalog = json.load(f)
    except Exception:
        product_catalog = {}

    try:
        with open(faq_path, "r", encoding="utf-8") as f:
            faqs = json.load(f)
    except Exception:
        faqs = {}

    tech_docs = ""
    try:
        with open(tech_docs_path, "r", encoding="utf-8") as f:
            tech_docs = f.read()
    except Exception:
        tech_docs = ""

    customer_conversations = []
    try:
        with open(conversations_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    customer_conversations.append(json.loads(line))
    except Exception:
        customer_conversations = []

    knowledge_base = {
        "product_catalog": product_catalog,
        "faqs": faqs,
        "tech_docs": tech_docs,
        "customer_conversations": customer_conversations,
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
    """Process a customer support query with monitoring and memory management"""
    start_time = time.time()
    conversation_id = query.conversation_id or f"conv-{datetime.utcnow().strftime('%Y%m%d%H%M%S')}"
    
    try:
        if not agent_orchestrator:
            raise HTTPException(status_code=500, detail="System not initialized properly")
        
        # Get or create conversation memory
        conversation = conversation_manager.get_or_create_conversation(conversation_id)
        
        # Add user query to conversation history
        conversation.add_turn("user", query.query)
        
        # Process query
        result = await agent_orchestrator.process_query(query.query, conversation_id)
        
        # Add assistant response to conversation history
        conversation.add_turn("assistant", result.get("response", ""), agent=result.get("agent", "unknown"))
        
        # Log metrics
        response_time = time.time() - start_time
        agent_metrics.log_query(
            agent=result.get("agent", "unknown"),
            success=True,
            response_time=response_time
        )
        
        # Save to database if available
        if db_available:
            try:
                db_manager = get_db_manager()
                db_manager.save_conversation({
                    "id": conversation_id,
                    "customer_id": conversation.metadata.get("customer_id"),
                    "agent_type": result.get("agent"),
                    "turns": conversation.get_turns(),
                    "conversation_metadata": conversation.metadata,  # Use correct field name
                    "status": "active"
                })
                db_manager.log_agent_action(
                    conversation_id=conversation_id,
                    agent_name=result.get("agent", "unknown"),
                    action_type="query_response",
                    action_details={"query": query.query, "response_time": response_time},
                    success=True
                )
            except Exception as db_error:
                logger.warning(f"Failed to save to database: {db_error}")
        
        return result
    except Exception as e:
        response_time = time.time() - start_time
        agent_metrics.log_query(
            agent="unknown",
            success=False,
            response_time=response_time,
            error=str(e)
        )
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


# ==================== PHASE 1: New Endpoints ====================

@app.get("/api/metrics")
async def get_metrics():
    """Get system metrics and performance data"""
    return {
        "status": "healthy",
        "metrics": agent_metrics.get_metrics(),
        "timestamp": datetime.utcnow().isoformat() + "Z"
    }


@app.get("/api/health/detailed")
async def detailed_health_check():
    """Comprehensive health check including all system components"""
    # Create a fresh health checker for this request
    from monitoring import HealthCheck
    health_checker = HealthCheck()
    
    # LLM health check
    def check_llm():
        return llm is not None and getattr(llm, "available", False)
    health_checker.register_component("llm", check_llm)
    
    # Agent orchestrator health check
    def check_orchestrator():
        return agent_orchestrator is not None
    health_checker.register_component("orchestrator", check_orchestrator)
    
    # Knowledge base health check
    def check_knowledge_base():
        return knowledge_base is not None and len(knowledge_base) > 0
    health_checker.register_component("knowledge_base", check_knowledge_base)
    
    # Database health check (if available)
    if db_available:
        def check_database():
            try:
                db_manager = get_db_manager()
                return db_manager is not None
            except:
                return False
        health_checker.register_component("database", check_database)
    
    return health_checker.check_health()


@app.get("/api/conversations/{conversation_id}")
async def get_conversation(conversation_id: str):
    """Get conversation history by ID"""
    conversation = conversation_manager.get_conversation(conversation_id)
    if conversation is None:
        raise HTTPException(status_code=404, detail="Conversation not found")
    
    return {
        "conversation_id": conversation_id,
        "summary": conversation.get_summary(),
        "turns": conversation.get_turns(),
        "metadata": conversation.metadata
    }


@app.get("/api/conversations")
async def list_conversations():
    """List all active conversations"""
    return {
        "conversations": conversation_manager.get_all_conversations(),
        "stats": conversation_manager.get_stats()
    }


@app.delete("/api/conversations/{conversation_id}")
async def delete_conversation(conversation_id: str):
    """Delete a conversation"""
    success = conversation_manager.delete_conversation(conversation_id)
    if not success:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return {"status": "deleted", "conversation_id": conversation_id}


@app.get("/api/tools")
async def list_tools():
    """List all available tools in the registry"""
    return {
        "tools": tool_registry.list_tools(),
        "execution_history": tool_registry.execution_history[-10:]  # Last 10 executions
    }


@app.post("/api/tools/{tool_name}/execute")
async def execute_tool(tool_name: str, request: Request):
    """Execute a tool directly (for testing/admin purposes)"""
    try:
        params = await request.json()
        result = tool_registry.execute(tool_name, **params)
        
        # Log tool usage
        agent_metrics.log_tool_usage(tool_name, result.get("success", False))
        
        return result
    except Exception as e:
        logger.error(f"Error executing tool {tool_name}: {e}")
        agent_metrics.log_tool_usage(tool_name, False)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/performance/operations")
async def get_performance_operations():
    """Get recent operation performance data"""
    return {
        "active_operations": performance_monitor.get_active_operations(),
        "recent_operations": performance_monitor.get_recent_operations(20)
    }


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


# ==================== PHASE 2: Advanced Features ====================

# ReAct Pattern Endpoints
@app.post("/api/query/react")
async def process_query_with_react(query: CustomerQuery):
    """Process query using ReAct pattern (Reason-Act-Observe)"""
    if not react_available or not agent_orchestrator:
        raise HTTPException(status_code=503, detail="ReAct not available")
    
    try:
        conversation_id = query.conversation_id or f"react-conv-{datetime.utcnow().strftime('%Y%m%d%H%M%S')}"
        
        # Create ReAct agent
        react_agent = ReActAgent(
            name="ReActProductAgent",
            llm_utils=llm,
            tool_registry=tool_registry,
            max_iterations=3
        )
        
        # Process with ReAct
        result = await react_agent.process_query_with_react(query.query)
        
        # Log metrics
        agent_metrics.log_query(
            agent="react_agent",
            success=result.get("success", False),
            response_time=0.5  # Placeholder
        )
        
        return {
            "response": result.get("final_answer", result.get("error", "")),
            "reasoning_trace": result.get("reasoning_process", []),
            "iterations": result.get("iterations", 0),
            "agent": "ReActAgent",
            "conversation_id": conversation_id,
            "method": "react"
        }
    except Exception as e:
        logger.error(f"ReAct query failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Tool Integrations Endpoints
@app.get("/api/integrations/status")
async def get_integrations_status():
    """Get status of all tool integrations"""
    if not integrations_available:
        return {"error": "Integrations not available"}
    
    return {
        "integrations": integration_factory.get_available_integrations(),
        "timestamp": datetime.utcnow().isoformat() + "Z"
    }


@app.post("/api/integrations/stripe/create-payment")
async def create_payment_intent(request: Request):
    """Create Stripe payment intent"""
    if not integrations_available:
        raise HTTPException(status_code=503, detail="Integrations not available")
    
    try:
        data = await request.json()
        stripe_integration = integration_factory.get_integration("stripe")
        
        result = stripe_integration.create_payment_intent(
            amount=data.get("amount"),
            currency=data.get("currency", "usd"),
            description=data.get("description")
        )
        
        return result
    except Exception as e:
        logger.error(f"Payment creation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/integrations/sendgrid/send-email")
async def send_email_via_sendgrid(request: Request):
    """Send email via SendGrid"""
    if not integrations_available:
        raise HTTPException(status_code=503, detail="Integrations not available")
    
    try:
        data = await request.json()
        sendgrid = integration_factory.get_integration("sendgrid")
        
        result = sendgrid.send_email(
            to_email=data.get("to"),
            subject=data.get("subject"),
            html_content=data.get("html_content")
        )
        
        return result
    except Exception as e:
        logger.error(f"Email sending failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/integrations/slack/notify")
async def send_slack_notification(request: Request):
    """Send Slack notification"""
    if not integrations_available:
        raise HTTPException(status_code=503, detail="Integrations not available")
    
    try:
        data = await request.json()
        slack = integration_factory.get_integration("slack")
        
        result = slack.post_notification(
            title=data.get("title"),
            message=data.get("message"),
            severity=data.get("severity", "info")
        )
        
        return result
    except Exception as e:
        logger.error(f"Slack notification failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Semantic Search Endpoints
@app.post("/api/search/semantic")
async def semantic_search(request: Request):
    """Perform semantic search across documents"""
    if not semantic_search_available:
        raise HTTPException(status_code=503, detail="Semantic search not available")
    
    try:
        data = await request.json()
        query = data.get("query")
        collection = data.get("collection", "documents")
        top_k = data.get("top_k", 5)
        
        # Ensure collection exists
        semantic_search_engine.create_collection(collection)
        
        result = semantic_search_engine.search(collection, query, top_k)
        
        return result
    except Exception as e:
        logger.error(f"Semantic search failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/search/hybrid")
async def hybrid_search(request: Request):
    """Perform hybrid semantic + keyword search"""
    if not semantic_search_available:
        raise HTTPException(status_code=503, detail="Semantic search not available")
    
    try:
        data = await request.json()
        query = data.get("query")
        keywords = data.get("keywords", [])
        collection = data.get("collection", "documents")
        top_k = data.get("top_k", 5)
        
        semantic_search_engine.create_collection(collection)
        
        result = semantic_search_engine.hybrid_search(
            collection, query, keywords, top_k
        )
        
        return result
    except Exception as e:
        logger.error(f"Hybrid search failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Admin Analytics Endpoints
@app.get("/api/admin/dashboard")
async def get_admin_dashboard():
    """Get admin dashboard summary"""
    if not analytics_available:
        return {"error": "Analytics not available"}
    
    return admin_analytics.get_dashboard_summary()


@app.get("/api/admin/agents/{agent}/analytics")
async def get_agent_analytics(agent: str):
    """Get analytics for specific agent"""
    if not analytics_available:
        return {"error": "Analytics not available"}
    
    return admin_analytics.get_agent_analytics(agent)


@app.get("/api/admin/satisfaction")
async def get_satisfaction_report(days: int = 7):
    """Get customer satisfaction report"""
    if not analytics_available:
        return {"error": "Analytics not available"}
    
    return admin_analytics.get_satisfaction_report(days)


@app.post("/api/admin/satisfaction/record")
async def record_satisfaction(request: Request):
    """Record customer satisfaction feedback"""
    if not analytics_available:
        raise HTTPException(status_code=503, detail="Analytics not available")
    
    try:
        data = await request.json()
        admin_analytics.record_satisfaction(
            rating=data.get("rating"),
            feedback=data.get("feedback"),
            agent=data.get("agent")
        )
        return {"success": True, "message": "Satisfaction recorded"}
    except Exception as e:
        logger.error(f"Failed to record satisfaction: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/admin/errors")
async def get_error_report():
    """Get error report and trending issues"""
    if not analytics_available:
        return {"error": "Analytics not available"}
    
    return admin_analytics.get_error_report()


@app.get("/api/admin/trends")
async def get_hourly_trends(hours: int = 24):
    """Get hourly query trends"""
    if not analytics_available:
        return {"error": "Analytics not available"}
    
    return admin_analytics.get_hourly_trend(hours)


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
    logger.info("Starting up Support Agent Orchestrator - Phase 2 Enhanced")
    logger.info(f"ReAct available: {react_available}")
    logger.info(f"Tool integrations available: {integrations_available}")
    logger.info(f"Semantic search available: {semantic_search_available}")
    logger.info(f"Admin analytics available: {analytics_available}")


# Application shutdown event
@app.on_event("shutdown")
async def shutdown_event():
    logger.info("Shutting down Support Agent Orchestrator")
    if semantic_search_available:
        semantic_search_engine.persist()


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


# ======================================
# PHASE 3: ENTERPRISE FEATURES ENDPOINTS
# ======================================

# -------- 3.1: AGENT COLLABORATION --------

@app.post("/api/collaboration/register-agent")
async def register_collaboration_agent(agent_id: str, agent_name: str, specialty: str, expertise_score: float = 0.7):
    """Register an agent for collaboration"""
    if not collaboration_available:
        raise HTTPException(status_code=503, detail="Collaboration not available")
    
    agent = agent_collaboration_hub.register_agent(agent_id, agent_name, specialty, expertise_score)
    return {"success": True, "agent": agent}


@app.post("/api/collaboration/initiate")
async def initiate_collaboration(
    session_id: str,
    topic: str,
    agent_ids: Optional[List[str]] = None,
    strategy: str = "majority"
):
    """Start a multi-agent collaboration session"""
    if not collaboration_available:
        raise HTTPException(status_code=503, detail="Collaboration not available")
    
    from agent_collaboration import ConsensusStrategy
    session = await agent_collaboration_hub.initiate_collaboration(
        session_id, topic, agent_ids, ConsensusStrategy(strategy)
    )
    return {"success": True, "session": session}


@app.post("/api/collaboration/opinion")
async def add_agent_opinion(
    session_id: str,
    agent_id: str,
    opinion: str,
    confidence: float,
    reasoning: str
):
    """Add an agent's opinion to collaboration"""
    if not collaboration_available:
        raise HTTPException(status_code=503, detail="Collaboration not available")
    
    agent_opinion = await agent_collaboration_hub.add_agent_opinion(
        session_id, agent_id, opinion, confidence, reasoning
    )
    return {"success": True, "opinion": agent_opinion.to_dict()}


@app.post("/api/collaboration/consensus")
async def reach_consensus(session_id: str):
    """Reach consensus from collected opinions"""
    if not collaboration_available:
        raise HTTPException(status_code=503, detail="Collaboration not available")
    
    consensus = await agent_collaboration_hub.reach_consensus(session_id)
    return {"success": True, "consensus": consensus.to_dict()}


@app.post("/api/collaboration/escalate")
async def escalate_issue(issue: str, severity: str = "medium", specialist_required: Optional[str] = None):
    """Escalate an issue"""
    if not collaboration_available:
        raise HTTPException(status_code=503, detail="Collaboration not available")
    
    escalation = await agent_collaboration_hub.escalate_issue(issue, severity, specialist_required)
    return {"success": True, "escalation": escalation}


@app.get("/api/collaboration/metrics")
async def get_collaboration_metrics():
    """Get collaboration metrics"""
    if not collaboration_available:
        raise HTTPException(status_code=503, detail="Collaboration not available")
    
    metrics = agent_collaboration_hub.get_collaboration_metrics()
    return {"success": True, "metrics": metrics}


# -------- 3.2: DYNAMIC AGENT CREATION --------

@app.get("/api/agents/templates")
async def list_agent_templates():
    """List available agent templates"""
    if not dynamic_agents_available:
        raise HTTPException(status_code=503, detail="Dynamic agents not available")
    
    templates = dynamic_agent_factory.list_templates()
    return {"success": True, "templates": templates}


@app.get("/api/agents/tools")
async def list_available_tools():
    """List all available tools for agent binding"""
    if not dynamic_agents_available:
        raise HTTPException(status_code=503, detail="Dynamic agents not available")
    
    tools = dynamic_agent_factory.list_available_tools()
    return {"success": True, "tools": tools}


@app.post("/api/agents/from-template")
async def create_agent_from_template(
    agent_id: str,
    name: str,
    template_name: str,
    created_by: str = "system",
    custom_tools: Optional[List[str]] = None
):
    """Create an agent from a template"""
    if not dynamic_agents_available:
        raise HTTPException(status_code=503, detail="Dynamic agents not available")
    
    agent = await dynamic_agent_factory.create_agent_from_template(
        agent_id, name, template_name, created_by, custom_tools
    )
    return {"success": True, "agent": agent.to_dict()}


@app.post("/api/agents/custom")
async def create_custom_agent(
    agent_id: str,
    name: str,
    description: str,
    personality: str,
    expertise_level: float,
    tools: List[str],
    system_prompt: str,
    response_style: str = "helpful",
    temperature: float = 0.5,
    max_turns: int = 10
):
    """Create a fully custom agent"""
    if not dynamic_agents_available:
        raise HTTPException(status_code=503, detail="Dynamic agents not available")
    
    agent = await dynamic_agent_factory.create_custom_agent(
        agent_id, name, description, personality, expertise_level,
        tools, system_prompt, response_style, temperature, max_turns
    )
    return {"success": True, "agent": agent.to_dict()}


@app.get("/api/agents/{agent_id}")
async def get_agent(agent_id: str):
    """Get agent configuration"""
    if not dynamic_agents_available:
        raise HTTPException(status_code=503, detail="Dynamic agents not available")
    
    agent = dynamic_agent_factory.get_agent(agent_id)
    if not agent:
        raise HTTPException(status_code=404, detail="Agent not found")
    
    return {"success": True, "agent": agent.to_dict()}


@app.get("/api/agents")
async def list_agents(agent_type: Optional[str] = None):
    """List all agents or filter by type"""
    if not dynamic_agents_available:
        raise HTTPException(status_code=503, detail="Dynamic agents not available")
    
    agents = dynamic_agent_factory.list_agents(agent_type)
    return {"success": True, "agents": [a.to_dict() for a in agents]}


@app.post("/api/agents/{agent_id}/activate")
async def activate_agent(agent_id: str):
    """Activate an agent"""
    if not dynamic_agents_available:
        raise HTTPException(status_code=503, detail="Dynamic agents not available")
    
    agent = await dynamic_agent_factory.activate_agent(agent_id)
    return {"success": True, "agent": agent.to_dict()}


@app.post("/api/agents/{agent_id}/deactivate")
async def deactivate_agent(agent_id: str):
    """Deactivate an agent"""
    if not dynamic_agents_available:
        raise HTTPException(status_code=503, detail="Dynamic agents not available")
    
    agent = await dynamic_agent_factory.deactivate_agent(agent_id)
    return {"success": True, "agent": agent.to_dict()}


@app.get("/api/agents/factory/metrics")
async def get_factory_metrics():
    """Get agent factory metrics"""
    if not dynamic_agents_available:
        raise HTTPException(status_code=503, detail="Dynamic agents not available")
    
    metrics = dynamic_agent_factory.get_factory_metrics()
    return {"success": True, "metrics": metrics}


# -------- 3.3: ADVANCED ANALYTICS --------

@app.post("/api/analytics/satisfaction/record")
async def record_satisfaction(
    user_id: str,
    agent_id: str,
    rating: int,
    comment: str,
    resolution_time_seconds: int,
    issue_complexity: str = "medium",
    category: str = "general"
):
    """Record user satisfaction metric"""
    if not advanced_analytics_available:
        raise HTTPException(status_code=503, detail="Advanced analytics not available")
    
    metric = await advanced_analytics_engine.record_satisfaction(
        user_id, agent_id, rating, comment, resolution_time_seconds,
        issue_complexity, category
    )
    return {"success": True, "metric": metric.to_dict()}


@app.get("/api/analytics/agent/{agent_id}/effectiveness")
async def get_agent_effectiveness(agent_id: str, start_date: Optional[str] = None, end_date: Optional[str] = None):
    """Get agent effectiveness metrics"""
    if not advanced_analytics_available:
        raise HTTPException(status_code=503, detail="Advanced analytics not available")
    
    effectiveness = await advanced_analytics_engine.calculate_agent_effectiveness(agent_id, start_date, end_date)
    if not effectiveness:
        raise HTTPException(status_code=404, detail="No metrics found for agent")
    
    return {"success": True, "effectiveness": effectiveness.to_dict()}


@app.post("/api/analytics/revenue/track")
async def track_revenue_impact(
    agent_id: str,
    revenue_generated: float,
    orders_facilitated: int,
    upsells_completed: int = 0,
    cost_of_operation: float = 10.0,
    period: str = "daily"
):
    """Track revenue impact of an agent"""
    if not advanced_analytics_available:
        raise HTTPException(status_code=503, detail="Advanced analytics not available")
    
    impact = await advanced_analytics_engine.track_revenue_impact(
        agent_id, revenue_generated, orders_facilitated, upsells_completed,
        cost_of_operation, period
    )
    return {"success": True, "impact": impact.to_dict()}


@app.post("/api/analytics/churn/predict")
async def predict_churn(customer_id: str):
    """Predict customer churn probability"""
    if not advanced_analytics_available:
        raise HTTPException(status_code=503, detail="Advanced analytics not available")
    
    prediction = await advanced_analytics_engine.predict_churn(customer_id)
    if not prediction:
        raise HTTPException(status_code=404, detail="No customer history found")
    
    return {"success": True, "prediction": {
        "customer_id": prediction.customer_id,
        "churn_probability": prediction.churn_probability,
        "risk_level": prediction.risk_level,
        "primary_factors": prediction.primary_factors,
        "recommended_actions": prediction.recommended_actions
    }}


@app.get("/api/analytics/cohort")
async def get_cohort_analysis(start_date: str, end_date: str):
    """Get cohort analysis"""
    if not advanced_analytics_available:
        raise HTTPException(status_code=503, detail="Advanced analytics not available")
    
    analysis = await advanced_analytics_engine.get_cohort_analysis(start_date, end_date)
    return {"success": True, "cohort_analysis": analysis}


@app.get("/api/analytics/trends")
async def get_trend_analysis(days: int = 30):
    """Get satisfaction trends"""
    if not advanced_analytics_available:
        raise HTTPException(status_code=503, detail="Advanced analytics not available")
    
    trends = await advanced_analytics_engine.get_trend_analysis(days)
    return {"success": True, "trends": trends}


@app.get("/api/analytics/summary")
async def get_analytics_summary():
    """Get comprehensive analytics summary"""
    if not advanced_analytics_available:
        raise HTTPException(status_code=503, detail="Advanced analytics not available")
    
    summary = advanced_analytics_engine.get_analytics_summary()
    return {"success": True, "summary": summary}


# -------- 3.4: REAL-TIME STREAMING --------

@app.post("/api/stream/create")
async def create_stream(query: str, agent_id: str, conversation_id: str):
    """Create a real-time streaming session"""
    if not streaming_available:
        raise HTTPException(status_code=503, detail="Streaming not available")
    
    stream_id = await realtime_streaming_engine.create_stream(query, agent_id, conversation_id)
    return {"success": True, "stream_id": stream_id}


@app.get("/api/stream/{stream_id}/events")
async def get_stream_events(stream_id: str):
    """Get all events for a stream"""
    if not streaming_available:
        raise HTTPException(status_code=503, detail="Streaming not available")
    
    events = realtime_streaming_engine.get_stream_events(stream_id)
    return {"success": True, "events": [e.to_json() for e in events]}


@app.get("/api/stream/{stream_id}/info")
async def get_stream_info(stream_id: str):
    """Get stream information"""
    if not streaming_available:
        raise HTTPException(status_code=503, detail="Streaming not available")
    
    info = realtime_streaming_engine.get_stream_info(stream_id)
    if not info:
        raise HTTPException(status_code=404, detail="Stream not found")
    
    return {"success": True, "info": info}


@app.get("/api/streaming/metrics")
async def get_streaming_metrics():
    """Get real-time streaming metrics"""
    if not streaming_available:
        raise HTTPException(status_code=503, detail="Streaming not available")
    
    metrics = realtime_streaming_engine.get_streaming_metrics()
    return {"success": True, "metrics": metrics}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
