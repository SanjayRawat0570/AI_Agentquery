import asyncio
import json
import os
from agent_implementations import LLMUtils, AgentOrchestrator

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")


def load_json(fname):
    path = os.path.join(DATA_DIR, fname)
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


async def main():
    # Create a disabled LLMUtils (no base_url) so agents use fallbacks
    llm = LLMUtils(base_url=None, model_name="")

    # Load knowledge base files
    product_catalog = load_json("product_catalog.json")
    faqs = load_json("faq.json")
    tech_docs_path = os.path.join(DATA_DIR, "tech_documentation.md")
    tech_docs = ""
    if os.path.exists(tech_docs_path):
        with open(tech_docs_path, "r", encoding="utf-8") as f:
            tech_docs = f.read()

    knowledge_base = {
        "product_catalog": product_catalog,
        "faqs": faqs,
        "tech_docs": tech_docs,
        "customer_conversations": [],
    }

    # No vector DB available in this lightweight demo
    vector_db = {"products": None, "technical": None, "conversations": None}

    orchestrator = AgentOrchestrator(llm, knowledge_base, vector_db)

    test_queries = [
        ("What is the monthly price for CM-Pro and what features does it include?", None),
        ("We're getting error E1234 when calling the API. What should we check?", None),
        ("Compare cm-pro vs cm-enterprise; also what's the monthly price for cm-enterprise?", None),
        ("How many user slots are available on ACC-1111?", None),
    ]

    for q, cid in test_queries:
        print("\n---\nQuery:\n", q)
        res = await orchestrator.process_query(q, cid)
        print("Response (agent=", res.get("agent"), "):\n", res.get("response"))

    # Demonstrate mid-session addition
    print("\n---\nAdding Account Management Agent mid-session via orchestrator API")
    add_result = orchestrator.add_account_management_agent()
    print("Add result:", add_result)

    # Now test an account query again
    q = "How many user slots are available on ACC-1111?"
    print("\nQuery after adding account agent:\n", q)
    res = await orchestrator.process_query(q, None)
    print("Response (agent=", res.get("agent"), "):\n", res.get("response"))


if __name__ == "__main__":
    asyncio.run(main())
