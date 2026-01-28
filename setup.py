import argparse
import logging
import os
import json

from agent_implementations import AgentOrchestrator, LLMUtils

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def setup_environment():
    """Set up the environment for the agent orchestrator"""
    # Create necessary directories
    os.makedirs("data", exist_ok=True)
    os.makedirs("chroma_db", exist_ok=True)

    # Try to initialize DataManager (requires chromadb/langchain). If unavailable,
    # fall back to loading KB files from the data directory and set vector_db to None
    knowledge_base = None
    vector_db = None
    try:
        from data_utils import DataManager

        # Initialize data manager
        data_manager = DataManager()

        # Load knowledge base
        logger.info("Loading knowledge base via DataManager...")
        knowledge_base = data_manager.load_knowledge_base()

        # Prepare vector database
        logger.info("Preparing vector database via DataManager...")
        vector_db = data_manager.prepare_vector_db(knowledge_base)
    except Exception as e:
        logger.warning(
            f"DataManager unavailable or failed to initialize: {e}.\n"
            "Falling back to loading local data files without vector DB."
        )
        data_dir = os.getenv("DATA_DIR", "data")
        try:
            with open(os.path.join(data_dir, "product_catalog.json"), "r", encoding="utf-8") as f:
                product_catalog = json.load(f)
        except Exception:
            product_catalog = {}

        try:
            with open(os.path.join(data_dir, "faq.json"), "r", encoding="utf-8") as f:
                faqs = json.load(f)
        except Exception:
            faqs = {}

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

    # Initialize LLM utils
    ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    model_name = os.getenv("MODEL_NAME", "llama3:8b")
    llm_utils = LLMUtils(ollama_base_url, model_name)

    # Initialize agent orchestrator
    logger.info("Initializing agent orchestrator...")
    agent_orchestrator = AgentOrchestrator(llm_utils, knowledge_base, vector_db)

    logger.info("Environment setup complete")
    return agent_orchestrator


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Set up the TechSolutions Agent Orchestrator environment"
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Reset the environment (delete existing data)",
    )

    args = parser.parse_args()

    if args.reset:
        logger.info("Resetting environment...")
        import shutil

        if os.path.exists("chroma_db"):
            shutil.rmtree("chroma_db")

    setup_environment()