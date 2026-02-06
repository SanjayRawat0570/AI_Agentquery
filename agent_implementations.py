import asyncio
import json
import logging
import os
from typing import Any, Dict, List, Optional, Tuple

import requests
try:
    from chromadb.api import Collection
except Exception:
    # chromadb may not be installed in the environment where we're running quick checks.
    # Provide a fallback placeholder for typing so module import won't fail.
    Collection = object

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# ==================== PHASE 1.1: Tool/Function Calling Framework ====================
class ToolRegistry:
    """Registry for agent-callable tools with execution tracking"""
    
    def __init__(self):
        self.tools: Dict[str, Dict[str, Any]] = {}
        self.execution_history: List[Dict[str, Any]] = []
        logger.info("ToolRegistry initialized")
    
    def register(self, name: str, func: callable, description: str, params: Optional[Dict] = None):
        """Register a tool with its function, description, and parameters"""
        self.tools[name] = {
            "func": func,
            "description": description,
            "params": params or {}
        }
        logger.info(f"Registered tool: {name}")
    
    def execute(self, tool_name: str, **kwargs) -> Dict[str, Any]:
        """Execute a tool and log the action"""
        if tool_name not in self.tools:
            error_msg = f"Tool {tool_name} not found in registry"
            logger.error(error_msg)
            return {"success": False, "error": error_msg}
        
        try:
            result = self.tools[tool_name]["func"](**kwargs)
            self.execution_history.append({
                "tool": tool_name,
                "params": kwargs,
                "result": result,
                "success": True,
                "timestamp": self._get_timestamp()
            })
            logger.info(f"Tool executed successfully: {tool_name}")
            return {"success": True, "result": result}
        except Exception as e:
            error_msg = str(e)
            self.execution_history.append({
                "tool": tool_name,
                "params": kwargs,
                "error": error_msg,
                "success": False,
                "timestamp": self._get_timestamp()
            })
            logger.error(f"Tool execution failed: {tool_name} - {error_msg}")
            return {"success": False, "error": error_msg}
    
    def get_tool_description(self, tool_name: str) -> Optional[str]:
        """Get description of a specific tool"""
        if tool_name in self.tools:
            return self.tools[tool_name]["description"]
        return None
    
    def list_tools(self) -> List[Dict[str, str]]:
        """List all available tools"""
        return [
            {"name": name, "description": tool["description"]}
            for name, tool in self.tools.items()
        ]
    
    def _get_timestamp(self) -> str:
        """Get current timestamp in ISO format"""
        from datetime import datetime
        return datetime.utcnow().isoformat() + "Z"


# Example tool implementations
def get_customer_info(customer_id: str) -> Dict[str, Any]:
    """Retrieve customer information from mock database"""
    # Mock implementation - in production, this would query a real database
    mock_customers = {
        "CUST-001": {
            "id": "CUST-001",
            "name": "John Doe",
            "email": "john@example.com",
            "account_id": "ACC-1111",
            "subscription": "cm-pro",
            "status": "active"
        },
        "CUST-002": {
            "id": "CUST-002",
            "name": "Jane Smith",
            "email": "jane@example.com",
            "account_id": "ACC-2222",
            "subscription": "cm-enterprise",
            "status": "active"
        }
    }
    return mock_customers.get(customer_id, {"error": "Customer not found"})


def check_inventory(product_id: str) -> Dict[str, Any]:
    """Check product inventory status"""
    # Mock implementation
    mock_inventory = {
        "PROD-CM-PRO": {"in_stock": True, "quantity": 1000, "status": "Available"},
        "PROD-CM-ENT": {"in_stock": True, "quantity": 500, "status": "Available"},
        "PROD-CM-STR": {"in_stock": True, "quantity": 2000, "status": "Available"}
    }
    return mock_inventory.get(product_id, {"in_stock": False, "status": "Not found"})


def create_support_ticket(issue: str, priority: str = "medium", customer_id: Optional[str] = None) -> Dict[str, Any]:
    """Create a support ticket"""
    from datetime import datetime
    ticket_id = f"TICKET-{datetime.now().strftime('%Y%m%d%H%M%S')}"
    return {
        "ticket_id": ticket_id,
        "issue": issue,
        "priority": priority,
        "customer_id": customer_id,
        "status": "open",
        "created_at": datetime.utcnow().isoformat() + "Z"
    }


def send_email(to: str, subject: str, body: str) -> Dict[str, Any]:
    """Send email notification (mock implementation)"""
    logger.info(f"[MOCK EMAIL] To: {to}, Subject: {subject}")
    return {
        "sent": True,
        "to": to,
        "subject": subject,
        "message_id": f"MSG-{hash(to + subject) % 10000}"
    }


# Initialize global tool registry
tool_registry = ToolRegistry()

# Register default tools
tool_registry.register(
    "get_customer_info",
    get_customer_info,
    "Retrieve customer information including subscription and account details",
    {"customer_id": "string"}
)
tool_registry.register(
    "check_inventory",
    check_inventory,
    "Check product availability and inventory status",
    {"product_id": "string"}
)
tool_registry.register(
    "create_support_ticket",
    create_support_ticket,
    "Create a new support ticket for customer issues",
    {"issue": "string", "priority": "string", "customer_id": "string"}
)
tool_registry.register(
    "send_email",
    send_email,
    "Send email notifications to customers",
    {"to": "string", "subject": "string", "body": "string"}
)


# LLM utilities
class LLMUtils:
    def __init__(self, base_url: str, model_name: str):
        self.base_url = base_url
        self.model_name = model_name
        # Check connectivity to the LLM endpoint early and mark availability.
        self.available = True
        try:
            # quick health check with a short timeout
            if not self.base_url:
                raise RuntimeError("No base_url configured for LLM")
            health_url = f"{self.base_url}/api/generate"
            requests.head(health_url, timeout=1)
        except Exception as e:
            logger.warning(f"LLM endpoint not reachable during init: {e}")
            self.available = False

        logger.info(f"Initialized LLM interface with model: {model_name}; available={self.available}")

    def generate_response(
        self, prompt: str, system_prompt: Optional[str] = None
    ) -> str:
        """Generate a response from the LLM using Ollama API"""
        # If we previously detected the LLM as unavailable, still attempt a request
        # but use a longer timeout and a single retry to allow for transient slowness.
        if not getattr(self, "available", False):
            logger.warning("LLM previously marked unavailable; attempting one request before falling back.")

        url = f"{self.base_url}/api/generate"
        payload = {"model": self.model_name, "prompt": prompt, "stream": False}
        if system_prompt:
            payload["system"] = system_prompt

        # Try up to two attempts (initial + one retry) with a longer timeout to avoid ReadTimeout for slow models
        attempts = 2
        timeout_seconds = 30
        last_exception = None
        for attempt in range(1, attempts + 1):
            try:
                response = requests.post(url, json=payload, timeout=timeout_seconds)
                logger.info(f"LLM response status: {getattr(response, 'status_code', None)} (attempt {attempt})")
                logger.debug(getattr(response, 'text', '')[:1000])
                logger.info("********************************************")
                # Success: mark available and return
                self.available = True
                return response.json().get("response", "")
            except requests.exceptions.RequestException as e:
                logger.warning(f"LLM request attempt {attempt} failed: {e}")
                last_exception = e
                # small backoff before retrying
                if attempt < attempts:
                    import time

                    time.sleep(1)
                continue

        # After retries, mark as unavailable and return helpful fallback message
        logger.error(f"LLM requests failed after {attempts} attempts: {last_exception}")
        self.available = False
        return (
            f"The language model service is currently unreachable. I will attempt to answer using local knowledge. Error: {str(last_exception)}"
        )

    def check_availability(self, timeout: float = 2.0) -> bool:
        """Check whether the LLM endpoint is reachable and update availability flag."""
        if not self.base_url:
            self.available = False
            return False
        try:
            health_url = f"{self.base_url}/api/generate"
            requests.head(health_url, timeout=timeout)
            self.available = True
        except Exception as e:
            logger.warning(f"LLM health check failed: {e}")
            self.available = False
        return self.available


# ==================== PHASE 1.4: Enhanced Error Handling & Self-Correction ====================
class SelfCorrectingMixin:
    """Mixin to add self-correction capabilities to agents"""
    
    def is_valid_response(self, response: str, query: str = None) -> Tuple[bool, Optional[str]]:
        """
        Check if response meets quality criteria
        
        Returns:
            Tuple of (is_valid, reason_if_invalid)
        """
        # Length check
        if len(response.strip()) < 10:
            return False, "Response too short"
        
        # Check for error indicators
        error_keywords = ["error:", "failed:", "exception:", "cannot process"]
        if any(keyword in response.lower() for keyword in error_keywords):
            return False, "Response contains error indicators"
        
        # Check for unhelpful responses
        unhelpful_phrases = [
            "i don't know",
            "i cannot help",
            "i'm not sure",
            "no information available"
        ]
        response_lower = response.lower()
        if any(phrase in response_lower for phrase in unhelpful_phrases):
            # Only invalid if the entire response is unhelpful
            if len(response.strip()) < 100:
                return False, "Response appears unhelpful"
        
        return True, None
    
    async def self_correct(self, query: str, failed_response: str, reason: str) -> str:
        """
        Attempt to fix incorrect response
        
        Args:
            query: Original user query
            failed_response: The response that failed validation
            reason: Why the response failed
            
        Returns:
            Corrected response
        """
        correction_prompt = f"""
The previous response to a customer query was inadequate.

Customer Query: {query}

Previous Response: {failed_response}

Issue with Response: {reason}

Please provide a better, more helpful response to the customer's query.
Make sure the response is:
1. Relevant to the query
2. Detailed and informative (at least a few sentences)
3. Professional and helpful in tone
4. Actionable if appropriate
"""
        
        correction_system_prompt = """
You are a senior customer support specialist reviewing and improving responses.
Provide clear, helpful, and professional responses to customer queries.
If you don't have specific information, acknowledge this but provide related helpful information.
"""
        
        try:
            corrected_response = self.llm_utils.generate_response(
                correction_prompt,
                correction_system_prompt
            )
            logger.info("Response self-corrected successfully")
            return corrected_response
        except Exception as e:
            logger.error(f"Self-correction failed: {e}")
            # Return a generic helpful message
            return f"Thank you for your question about: {query}. I apologize, but I'm having difficulty providing a detailed response at the moment. Please let me know if you'd like to rephrase your question or if there's a specific aspect I can help you with."


# Base Agent class
class BaseAgent(SelfCorrectingMixin):
    def __init__(self, llm_utils: LLMUtils):
        self.llm_utils = llm_utils
        self.correction_attempts = 0
        self.max_correction_attempts = 2

    async def process(
        self, query: str, conversation_history: List[Dict[str, Any]] = None
    ) -> str:
        """Process a query and return a response"""
        raise NotImplementedError("Subclasses must implement this method")
    
    async def process_with_validation(self, query: str, conversation_history: List[Dict[str, Any]] = None) -> str:
        """
        Process query with built-in validation and self-correction
        
        This method wraps the standard process() method with validation and correction logic.
        """
        self.correction_attempts = 0
        
        # First attempt
        response = await self.process(query, conversation_history)
        is_valid, reason = self.is_valid_response(response, query)
        
        # If valid, return immediately
        if is_valid:
            return response
        
        # Attempt self-correction
        logger.warning(f"Invalid response detected: {reason}. Attempting self-correction.")
        
        while self.correction_attempts < self.max_correction_attempts:
            self.correction_attempts += 1
            try:
                corrected_response = await self.self_correct(query, response, reason)
                is_valid, reason = self.is_valid_response(corrected_response, query)
                
                if is_valid:
                    logger.info(f"Response corrected successfully after {self.correction_attempts} attempt(s)")
                    return corrected_response
                
                response = corrected_response
            except Exception as e:
                logger.error(f"Self-correction attempt {self.correction_attempts} failed: {e}")
                break
        
        # If all corrections fail, return the best attempt we have
        logger.warning(f"Could not produce valid response after {self.correction_attempts} corrections")
        return response


# Router Agent
class RouterAgent(BaseAgent):
    def __init__(self, llm_utils: LLMUtils):
        super().__init__(llm_utils)
        self.system_prompt = """
        You are a Router Agent for TechSolutions customer support. Your job is to:
        1. Understand the customer's query
        2. Classify the query into one of these categories:
           - Product: Questions about products, features, pricing, plans
           - Technical: Questions about errors, issues, troubleshooting
           - Billing: Questions about orders, invoices, payments, subscriptions
           - Account: Questions about user management, access, settings
           - General: General inquiries that don't fit other categories
        3. For multi-part queries, identify each part and its category
        
        Respond with JSON in this format:
        {
            "classification": "Product" or "Technical" or "Billing" or "Account" or "General",
            "confidence": 0.9, # between 0 and 1
            "requires_clarification": false, # true if query is too vague
            "clarification_question": "optional question if requires_clarification is true"
        }
        
        For multi-part queries, respond with:
        {
            "multi_part": true,
            "parts": [
                {
                    "query_part": "extracted part of the query",
                    "classification": "Product" or "Technical" or "Billing" or "Account" or "General"
                },
                ...
            ]
        }
        """

        # self.system_prompt = """
        # You are a Router Agent for TechSolutions customer support. Your job is to:
        # 1. Understand the customer's query
        # 2. Strictly format your response as JSON without any additional text
        # 3. Use exactly one of the valid classifications: Product, Technical, Billing, Account, General
        
        # Valid response formats:
        
        # Single query:
        # {
        #     "classification": "Product|Technical|Billing|Account|General",
        #     "confidence": 0.9,
        #     "requires_clarification": false,
        #     "clarification_question": ""
        # }
        
        # Multi-part query:
        # {
        #     "multi_part": true,
        #     "parts": [
        #         {
        #             "query_part": "text",
        #             "classification": "Product|Technical|Billing|Account|General"
        #         }
        #     ]
        # }
        
        # If uncertain, set requires_clarification to true and provide a question.
        # """

    # def process(
    #     self, query: str, conversation_history: List[Dict[str, Any]] = None
    # ) -> Dict[str, Any]:
    #     prompt = f"Customer query: {query}\n\nPlease classify this query according to the instructions."
    #     response = self.llm_utils.generate_response(prompt, self.system_prompt)

    #     try:
    #         # Parse the JSON response
    #         result = json.loads(response)
    #         return result
    #     except json.JSONDecodeError:
    #         logger.error(f"Failed to parse Router Agent response as JSON: {response}")
    #         return {
    #             "classification": "General",
    #             "confidence": 0.5,
    #             "requires_clarification": True,
    #             "clarification_question": "Could you please provide more details about your question?",
    #         }

    async def process(self, query: str, conversation_history: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        # Implement a robust, rule-based classifier as primary path.
        # Use LLM only optionally (if available) but default to keyword rules for stability.
        q = query.strip()

        # Detect obvious multi-part queries by separators
        parts = [p.strip() for p in __import__('re').split(r"[;\n\?]\s*", q) if p.strip()]
        if len(parts) > 1:
            out_parts = []
            for p in parts:
                cls = self._keyword_classify(p)
                out_parts.append({"query_part": p, "classification": cls})
            return {"multi_part": True, "parts": out_parts}

        # Single-part: classify using keywords
        classification = self._keyword_classify(q)
        result = {
            "classification": classification,
            "confidence": 0.85,
            "requires_clarification": False,
            "clarification_question": "",
        }
        return result

    def _clean_json_response(self, response: str) -> str:
        """Remove non-JSON content from the response"""
        # Remove markdown code blocks
        response = response.replace('```json', '').replace('```', '')
        # Extract first JSON object
        start = response.find('{')
        end = response.rfind('}') + 1
        return response[start:end] if start != -1 and end != 0 else response

    def _validate_response_structure(self, result: Dict) -> Dict:
        """Ensure response has required fields"""
        required_single = ["classification", "confidence", "requires_clarification"]
        required_multi = ["multi_part", "parts"]
        
        if "multi_part" in result:
            if not all(k in result for k in required_multi):
                raise ValueError("Invalid multi-part structure")
            for part in result.get("parts", []):
                if "query_part" not in part or "classification" not in part:
                    raise ValueError("Invalid part structure")
        else:
            if not all(k in result for k in required_single):
                raise ValueError("Missing required fields")
        logger.info(f"Valid response structure: {result}")
        logger.info("********************************************")       
        return result

    def _safe_fallback_response(self, query: str, error: Exception) -> Dict:
        """Create a safe fallback response when parsing fails"""
        logger.warning(f"Using fallback classification for query: {query}")
        # Use a lightweight keyword-based classifier as a fallback
        fallback_class = self._keyword_classify(query)
        return {
            "classification": fallback_class,
            "confidence": 0.6,
            "requires_clarification": False,
            "clarification_question": "",
            "parse_error": str(error),
        }

    def _keyword_classify(self, query: str) -> str:
        """Simple keyword-based classifier used as a fallback when LLM parsing fails."""
        q = query.lower()
        # Account ID detection: if an account id is present, classify as Account
        import re
        if re.search(r"acc-\d+", q):
            return "Account"
        # Detect multi-part queries
        # If user asks multiple questions separated by 'and', ';', 'also', treat as multi-part
        if any(sep in q for sep in [";", " and ", " also ", " then ", "\n"]):
            # Fallback to Product for multi-part by default
            return "Product"

        if any(k in q for k in ["price", "cost", "billing", "invoice", "order", "payment"]):
            return "Billing"
        if any(k in q for k in ["install", "error", "fail", "crash", "trouble", "issue", "bug", "timeout"]):
            return "Technical"
        if any(k in q for k in ["account", "login", "signup", "access", "users", "permission"]):
            return "Account"
        if any(k in q for k in ["feature", "compare", "pricing", "plan", "product", "catalog"]):
            return "Product"
        return "General"


# Product Specialist Agent
class ProductSpecialistAgent(BaseAgent):
    def __init__(
        self,
        llm_utils: LLMUtils,
        product_catalog: Dict[str, Any],
        faqs: Dict[str, Any],
        vector_db: Optional[Any],
    ):
        super().__init__(llm_utils)
        self.product_catalog = product_catalog
        self.faqs = faqs
        self.vector_db = vector_db
        self.system_prompt = """
        You are a Product Specialist Agent for TechSolutions customer support.
        You're an expert on TechSolutions products, features, pricing, and plans.
        
        When responding to customer queries:
        1. Be accurate and specific about product features and pricing
        2. Compare products when relevant to help customers choose
        3. Highlight benefits and use cases for specific products
        4. If you don't know something, say so rather than guessing
        
        Keep your responses friendly, concise, and focused on answering the customer's specific question.
        """

    async def _retrieve_relevant_information(self, query: str) -> str:
        """Retrieve relevant product information from the knowledge base"""
        try:
            # If vector DB available, use it (async thread)
            if self.vector_db:
                results = await asyncio.to_thread(self.vector_db.query, query_texts=[query], n_results=3)
                relevant_info = "\n\n".join(results.get("documents", [[]])[0])
                return relevant_info

            # Fallback: simple catalog search
            hits = []
            ql = query.lower()
            for p in self.product_catalog.get("products", []):
                if p.get("id", "").lower() in ql or p.get("name", "").lower() in ql:
                    hits.append(json.dumps(p, indent=2))
            # If no direct id/name match, do keyword match on features/description
            if not hits:
                for p in self.product_catalog.get("products", []):
                    text = " ".join([p.get("name", ""), p.get("description", "")]).lower()
                    if any(tok in text for tok in ql.split()[:5]):
                        hits.append(p.get("description", ""))

            return "\n\n".join(hits)
        except Exception as e:
            logger.error(f"Error retrieving information from vector database: {e}")
            return ""

    async def process(
        self, query: str, conversation_history: List[Dict[str, Any]] = None
    ) -> str:
        # Retrieve relevant information from the knowledge base (run blocking call in thread)
        relevant_info = await self._retrieve_relevant_information(query)

        logger.info(f"Relevant information: {relevant_info}")
        logger.info("********************************************")

        # Construct the prompt
        prompt = f"""
        Customer query: {query}
        
        Relevant information:
        {relevant_info}
        
        Please provide a helpful response based on this information.
        """
        # Detect product comparison requests (e.g., "compare X and Y", "X vs Y") and synthesize from catalog when possible
        import re

        cmp_match = re.search(r"compare\s+([\w\-\s]+)\s+(?:vs|and)\s+([\w\-\s]+)", query, re.I)
        if cmp_match:
            name_a = cmp_match.group(1).strip()
            name_b = cmp_match.group(2).strip()

            def find_product(name: str):
                nl = name.lower()
                for p in self.product_catalog.get("products", []):
                    if nl in p.get("id", "").lower() or nl in p.get("name", "").lower():
                        return p
                return None

            pa = find_product(name_a)
            pb = find_product(name_b)
            if pa or pb:
                # Build comparison table
                def summarize(p):
                    if not p:
                        return "Not found"
                    return (
                        f"Name: {p.get('name')}\nID: {p.get('id')}\nPrice monthly: {p.get('price', {}).get('monthly')}\n"
                        f"Price annual: {p.get('price', {}).get('annual')}\nFeatures:\n" + "\n".join([f"- {f.get('name')}: {f.get('description')}" for f in p.get('features', [])])
                    )

                comparison_text = f"Comparison: {name_a} vs {name_b}\n\nProduct A:\n{summarize(pa)}\n\nProduct B:\n{summarize(pb)}"
                # If an LLM is available, include the comparison in the prompt; otherwise return directly
                if self.llm_utils and getattr(self.llm_utils, "base_url", None):
                    prompt = prompt + "\n\n" + comparison_text
                    response = await asyncio.to_thread(self.llm_utils.generate_response, prompt, self.system_prompt)
                    return response
                return comparison_text

        # If an LLM is configured (llm_utils.base_url), use it; otherwise return a simple synthesized answer.
        if self.llm_utils and getattr(self.llm_utils, "base_url", None):
            response = await asyncio.to_thread(self.llm_utils.generate_response, prompt, self.system_prompt)
            return response
        # Fallback: synthesize a basic response from retrieved info
        if relevant_info:
            return f"Here is what I found about your product query:\n\n{relevant_info}"
        return "I'm sorry — I couldn't find product information matching your query. Could you provide the product name or ID?"


# Technical Support Agent
class TechnicalSupportAgent(BaseAgent):
    def __init__(self, llm_utils: LLMUtils, tech_docs: str, vector_db: Optional[Any]):
        super().__init__(llm_utils)
        self.tech_docs = tech_docs
        self.vector_db = vector_db
        self.system_prompt = """
        You are a Technical Support Agent for TechSolutions customer support.
        You're an expert in troubleshooting TechSolutions products and resolving technical issues.
        
        When responding to customer queries:
        1. Identify the specific issue or error described
        2. Provide step-by-step troubleshooting instructions
        3. Reference relevant documentation when applicable
        4. Suggest preventive measures for future reference
        
        Keep your responses clear, structured, and focused on resolving the customer's technical problem.
        """

    async def _retrieve_troubleshooting_info(self, query: str) -> str:
        """Retrieve relevant troubleshooting information (async)."""
        try:
            # Query the vector database for relevant information in a thread
            if self.vector_db:
                results = await asyncio.to_thread(self.vector_db.query, query_texts=[query], n_results=3)
            else:
                results = {"documents": [[self.tech_docs]]}

            logger.info(f"Retrieved troubleshooting information: {results}")
            logger.info("********************************************")

            # Extract and return the relevant information
            relevant_info = "\n\n".join(results.get("documents", [[]])[0])
            return relevant_info
        except Exception as e:
            logger.error(f"Error retrieving troubleshooting information: {e}")
            return ""


    async def _call_diagnostic_api(self, issue_description: str) -> Dict[str, Any]:
        """Call the diagnostic API for automated issue identification asynchronously"""
        try:
            # Run the blocking requests.post in a thread so it doesn't block the event loop
            response = await asyncio.to_thread(
                requests.post,
                "http://localhost:8000/api/diagnose",
                json={"description": issue_description},
            )
            response.raise_for_status()
            logger.info(f"Diagnostic API response: {response}")
            logger.info("********************************************")
            result = response.json()
            logger.info(f"Diagnostic API returned JSON: {result}")
            return result
        except Exception as e:
            logger.error(f"Error calling diagnostic API: {e}")
            return {}

    async def process(
        self, query: str, conversation_history: List[Dict[str, Any]] = None
    ) -> str:
        # Retrieve relevant troubleshooting information (async)
        relevant_info = await self._retrieve_troubleshooting_info(query)
        logger.info(f"Relevant troubleshooting information: {relevant_info}")
        logger.info("********************************************")

        # Get diagnostic suggestions asynchronously
        diagnostic_info = await self._call_diagnostic_api(query)
        logger.info(f"Diagnostic API response from process function: {diagnostic_info}")
        logger.info("********************************************")
        diagnostic_text = ""
        if diagnostic_info:
            solutions = diagnostic_info.get("solutions", [])
            solutions_text = "\n".join([f"- {solution}" for solution in solutions])
            diagnostic_text = f"""
            Diagnostic results:
            Issue: {diagnostic_info.get('name', 'Unknown issue')}
            Suggested solutions:
            {solutions_text}
            Documentation: {diagnostic_info.get('documentation_link', '')}
            """

        # Construct the prompt for the LLM
        prompt = f"""
        Customer query: {query}
        
        Relevant troubleshooting information:
        {relevant_info}
        
        {diagnostic_text}
        
        Please provide a helpful response to resolve this technical issue.
        """
        # Generate response using the LLM (run in thread to avoid blocking)
        response = await asyncio.to_thread(self.llm_utils.generate_response, prompt, self.system_prompt)
        return response



# Order/Billing Agent
class OrderBillingAgent(BaseAgent):
    def __init__(self, llm_utils: LLMUtils, product_catalog: Dict[str, Any]):
        super().__init__(llm_utils)
        self.product_catalog = product_catalog
        self.system_prompt = """
        You are an Order and Billing Agent for TechSolutions customer support.
        You're an expert in handling inquiries about orders, invoices, payments, and subscriptions.
        
        When responding to customer queries:
        1. Be precise about order status, payment information, and subscription details
        2. Explain billing charges clearly and transparently
        3. Outline available payment options and subscription changes when relevant
        4. Maintain a professional and reassuring tone
        
        Keep your responses clear, specific, and focused on addressing the customer's billing-related questions.
        """

    async def _get_order_details(self, order_id: str) -> Dict[str, Any]:
        """Retrieve order details from the Order API"""
        try:
            response = await asyncio.to_thread(
            requests.get, f"http://localhost:8000/api/orders/{order_id}"
        )
            response.raise_for_status()
            logger.info(f"Order details from the order API: {response}")
            logger.info("********************************************")
            return response.json()
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 404:
                logger.warning(f"Order not found: {order_id}")
                return {}
            else:
                logger.error(f"Error retrieving order details: {e}")
                return {}
        except Exception as e:
            logger.error(f"Error retrieving order details: {e}")
            return {}

    async def _get_account_details(self, account_id: str) -> Dict[str, Any]:
        """Retrieve account details from the Account API"""
        try:
            response = await asyncio.to_thread(
            requests.get, f"http://localhost:8000/api/accounts/{account_id}"
        )
            logger.info(f"Account details from the account API: {response}")
            logger.info("********************************************")
            response.raise_for_status()
            return response.json()
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 404:
                logger.warning(f"Account not found: {account_id}")
                return {}
            else:
                logger.error(f"Error retrieving account details: {e}")
                return {}
        except Exception as e:
            logger.error(f"Error retrieving account details: {e}")
            return {}

    async def process(
        self, query: str, conversation_history: List[Dict[str, Any]] = None
    ) -> str:
        # Extract order ID if present in the query
        order_id = None
        import re

        order_match = re.search(r"ORD-\d+", query)
        if order_match:
            order_id = order_match.group(0)
        
        logger.info(f"Order ID from the process function: {order_id}")
        logger.info("********************************************")

        # Extract account ID if present in the query
        account_id = None
        account_match = re.search(r"ACC-\d+", query)
        if account_match:
            account_id = account_match.group(0)

        logger.info(f"Account ID from the process function: {account_id}")
        logger.info("********************************************")

        # Retrieve order details if order ID is available
        order_details = {}
        if order_id:
            order_details = await self._get_order_details(order_id)

        logger.info(f"Order details from the process function: {order_details}")
        logger.info("********************************************")

        # Retrieve account details if account ID is available
        account_details = {}
        if account_id:
            account_details = await self._get_account_details(account_id)

        logger.info(f"Account details from the process function: {account_details}")
        logger.info("********************************************")

        # Construct the prompt with available information
        prompt = f"Customer query: {query}\n\n"

        if order_details:
            prompt += f"Order information:\n{json.dumps(order_details, indent=2)}\n\n"

        if account_details:
            prompt += (
                f"Account information:\n{json.dumps(account_details, indent=2)}\n\n"
            )

        # Add product pricing information if relevant
        if (
            "pricing" in query.lower()
            or "cost" in query.lower()
            or "price" in query.lower()
        ):
            products_info = json.dumps(
                self.product_catalog.get("products", []), indent=2
            )
            prompt += f"Product pricing information:\n{products_info}\n\n"

        prompt += "Please provide a helpful response to this billing or order question."

        # Generate response (run blocking LLM call in thread)
        response = await asyncio.to_thread(self.llm_utils.generate_response, prompt, self.system_prompt)
        logger.info(f"Billing Agent response: {response}")
        logger.info("********************************************")
        return response

# Account Management Agent
class AccountManagementAgent(BaseAgent):
    def __init__(self, llm_utils: LLMUtils):
        super().__init__(llm_utils)
        self.system_prompt = """
        You are an Account Management Agent for TechSolutions customer support.
        You are an expert in handling account queries, including user management, subscription details, and available user slots.
        
        When responding to customer queries:
        1. Confirm the action requested (e.g., adding users).
        2. Retrieve account details to check the current subscription tier and available user slots.
        3. Provide step-by-step instructions for adding new users.
        4. Offer additional suggestions if the account has reached its user limit.
        
        Keep your responses clear, structured, and detailed.
        """
    
    async def _get_account_info(self, account_id: str) -> Dict[str, Any]:
        """Retrieve account details using the Account API."""
        try:
            url = f"http://localhost:8000/api/accounts/{account_id}"
            response = await asyncio.to_thread(requests.get, url)
            response.raise_for_status()
            logger.info(f"Account API response: {response}")
            logger.info(f"Account API response status: {response.status_code}")
            logger.info(f"Raw account API response text: {response.text}")
            logger.info("********************************************")
            return response.json()
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 404:
                logger.warning(f"Account not found: {account_id}")
                return {}
            else:
                logger.error(f"Error retrieving account details: {e}")
                return {}
        except Exception as e:
            logger.error(f"Error retrieving account details: {e}")
            return {}
    
    async def process(
        self, query: str, conversation_history: List[Dict[str, Any]] = None
    ) -> str:
        """
        Process an account management query.
        For this scenario, we'll assume that the account id might be extracted from the query.
        If not, you could use a default value or query context.
        """
        import re

        # Extract an account id from the query (e.g., ACC-1111). This is just one approach.
        account_id = None
        account_match = re.search(r"ACC-\d+", query)
        if account_match:
            account_id = account_match.group(0)
        
        # For testing purposes, if no account id is provided, use a dummy account id.
        if not account_id:
            account_id = "ACC-1111"

        logger.info(f"Account ID extracted: {account_id}")

        # Retrieve account details from the account API (may return {} if API not reachable)
        account_info = await self._get_account_info(account_id)
        logger.info(f"Retrieved account info: {account_info}")

        # Intent parsing: view subscription, change subscription, verify permission
        ql = query.lower()

        # Check for subscription change intents
        import re
        change_match = re.search(r"(?:change|upgrade|downgrade|switch) (?:my )?(?:subscription|plan|tier) to (cm-[\w-]+)", ql)
        if change_match:
            requested_plan = change_match.group(1)
            # If account info available, simulate a subscription change request
            if account_info:
                old_plan = account_info.get("subscription", {}).get("plan", "unknown")
                # Here we simulate performing the change; in production you'd call the billing API
                response = (
                    f"I can help change the subscription for {account_id}.\n"
                    f"Current plan: {old_plan}. Requested plan: {requested_plan}.\n"
                    f"I have submitted a request to change the subscription. You will receive a confirmation email shortly."
                )
                return response
            else:
                return (
                    f"I couldn't reach the Account API to perform the change for {account_id}.\n"
                    f"To change the subscription manually, please: 1) Log in to the customer portal, 2) Go to Billing > Subscription, 3) Select the desired plan ({requested_plan}) and follow the prompts."
                )

        # Check for simple subscription info queries
        if any(tok in ql for tok in ["subscription", "plan", "tier", "pricing"]) and "change" not in ql:
            subscription = account_info.get("subscription", {})
            plan = subscription.get("plan", "unknown")
            status = subscription.get("status", "unknown")
            renewal = subscription.get("renewal_date", "unknown")
            payment = subscription.get("payment_method", "unknown")

            if account_info:
                return (
                    f"Account {account_id} subscription details:\n"
                    f"- Plan: {plan}\n"
                    f"- Status: {status}\n"
                    f"- Renewal date: {renewal}\n"
                    f"- Payment method: {payment}\n"
                    f"If you'd like to change plans, say 'Change my subscription to cm-enterprise' (for example)."
                )
            else:
                return (
                    f"I couldn't retrieve subscription details for {account_id} because the Account API is unavailable.\n"
                    f"You can check this in the portal at portal.techsolutions.example.com > Billing > Subscription."
                )

        # Permission verification intents
        perm_match = re.search(r"can (?:i|user) ([\w@.\-]+)? ?(perform|do|access) (.+)\??", ql)
        if perm_match:
            # Extract optional user and action
            user = perm_match.group(1) or "you"
            action = perm_match.group(3).strip()

            # Determine permission from roles in account_info
            if account_info:
                users = account_info.get("users", [])
                # If specific user requested, try to find role
                role = None
                if user != "you":
                    for u in users:
                        if u.get("email", "").lower() == user.lower():
                            role = u.get("role")
                            break
                else:
                    # assume requester is admin if not known
                    role = "admin"

                # Simple permission model
                role_perms = {
                    "admin": ["add user", "change subscription", "view billing", "manage settings", "deploy"],
                    "operator": ["deploy", "view billing", "run diagnostics"],
                    "viewer": ["view billing", "view dashboards"],
                    "billing": ["view billing", "manage invoices"],
                }

                allowed = False
                if role:
                    perms = role_perms.get(role.lower(), [])
                    # approximate check
                    if any(tok in action for tok in perms):
                        allowed = True

                if allowed:
                    return f"Yes — the user '{user}' with role '{role}' is allowed to {action}."
                else:
                    return f"No — user '{user}' with role '{role}' is not permitted to {action}. If you need this permission, please request an admin to update your role."
            else:
                return (
                    f"I can't verify permissions right now because the Account API is unavailable.\n"
                    f"If you need to check permissions, please visit Admin > User Management in the portal."
                )

        # Default behavior: provide account summary and guidance
        subscription = account_info.get("subscription", {})
        plan = subscription.get("plan", "unknown")
        current_user_count = len(account_info.get("users", []))
        return (
            f"Account summary for {account_id}:\n"
            f"- Plan: {plan}\n"
            f"- Number of users: {current_user_count}\n"
            f"If you need help changing subscriptions or managing users, ask something like 'Change my subscription to cm-enterprise' or 'How many user slots are available?'"
        )

# Orchestrator implementation
class AgentOrchestrator:
    def __init__(
        self,
        llm_utils: LLMUtils,
        knowledge_base: Dict[str, Any],
        vector_db: Dict[str, Any],
    ):
        self.llm_utils = llm_utils
        self.knowledge_base = knowledge_base
        self.vector_db = vector_db
        self.conversations = {}

        # Initialize agents
        self.router_agent = RouterAgent(llm_utils)
        self.product_agent = ProductSpecialistAgent(
            llm_utils,
            knowledge_base["product_catalog"],
            knowledge_base["faqs"],
            vector_db["products"],
        )
        self.technical_agent = TechnicalSupportAgent(
            llm_utils, knowledge_base["tech_docs"], vector_db["technical"]
        )
        self.billing_agent = OrderBillingAgent(
            llm_utils, knowledge_base["product_catalog"]
        )

        # Account Management Agent will be added during the mid-session challenge
        self.account_agent = AccountManagementAgent(llm_utils)

        logger.info("Agent Orchestrator initialized with all agents")

    async def process_query(
        self, query: str, conversation_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Process a customer query through the appropriate agent"""
        # Initialize conversation if new
        if conversation_id not in self.conversations:
            self.conversations[conversation_id] = []

        # Get conversation history
        conversation_history = self.conversations.get(conversation_id, [])

        # Route the query using the router agent
        routing_result = await self.router_agent.process(query, conversation_history)

        # Handle multi-part queries
        if routing_result.get("multi_part", False):
            responses = []
            for part in routing_result.get("parts", []):
                part_query = part.get("query_part")
                part_classification = part.get("classification")
                part_response = await self._process_single_query(
                    part_query, part_classification, conversation_history
                )
                responses.append(f"{part_response}")

            final_response = "\n\n".join(responses)
            agent_type = "multiple"
        else:
            # Handle single-part query
            classification = routing_result.get("classification", "General")

            # Check if clarification is needed
            if routing_result.get("requires_clarification", False):
                final_response = routing_result.get(
                    "clarification_question",
                    "Could you please provide more details about your question?",
                )
                agent_type = "router"
            else:
                final_response = await self._process_single_query(
                    query, classification, conversation_history
                )
                agent_type = classification.lower()

        # Update conversation history
        self.conversations[conversation_id].append(
            {"query": query, "response": final_response, "agent": agent_type}
        )

        return {
            "response": final_response,
            "agent": agent_type,
            "conversation_id": conversation_id or "new_conversation",
        }

    async def _process_single_query(
        self,
        query: str,
        classification: str,
        conversation_history: List[Dict[str, Any]],
    ) -> str:
        """Process a single-part query based on its classification"""
        if classification == "Product":
            logger.info(f"Processing product query, query: {query}")
            logger.info("********************************************")
            return await self.product_agent.process(query, conversation_history)
        elif classification == "Technical":
            logger.info(f"Processing technical query, query: {query}")
            logger.info("********************************************")
            return await self.technical_agent.process(query, conversation_history)
        elif classification == "Billing":
            logger.info(f"Processing billing query, query: {query}")
            logger.info("********************************************")
            return await self.billing_agent.process(query, conversation_history)
        elif classification == "Account":
            # Check if Account Management Agent is available (for mid-session challenge)
            if self.account_agent:
                logger.info(f"Processing account query, query: {query}")
                logger.info("********************************************")
                return await self.account_agent.process(query, conversation_history)
            else:
                # Fallback to billing agent if account agent not yet implemented
                fallback_response = await self.billing_agent.process(
                    query, conversation_history
                )
                return fallback_response
        else:
            # Default to a general response
            general_prompt = f"""
            Customer query: {query}
            
            Please provide a helpful and friendly general response to this query.
            """
            general_system_prompt = """
            You are a Customer Support Agent for TechSolutions.
            Provide helpful, friendly, and concise responses to general customer inquiries.
            If the query should be handled by a specialist agent, indicate which type of specialist would be appropriate.
            """
            return self.llm_utils.generate_response(
                general_prompt, general_system_prompt
            )

    # This method will be implemented during the mid-session challenge
    def add_account_management_agent(self):
        """Add the Account Management Agent (for mid-session challenge)"""
        # Dynamically add or replace the account management agent at runtime.
        try:
            self.account_agent = AccountManagementAgent(self.llm_utils)
            logger.info("Account Management Agent added during session")
            from datetime import datetime

            # Return timestamp for when the mid-session requirement was opened/added
            return {"added": True, "timestamp": datetime.utcnow().isoformat() + "Z"}
        except Exception as e:
            logger.error(f"Failed to add Account Management Agent: {e}")
            return {"added": False, "error": str(e)}
