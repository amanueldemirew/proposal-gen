"""
Proposal Generator API - LlamaIndex + Gemini

A LlamaIndex-based proposal generator that uses Q&A interactions to generate customized proposals.
"""

import os
import logging
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

# Load environment variables first
load_dotenv()

# Then import modules that use environment variables
from proposals.api.routes import router as proposal_router
from proposals.engine.llm import LLMConfig
from proposals.storage.session import SessionManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="Proposal Generator API",
    description="Generate customized proposals based on Q&A interactions",
    version="0.1.0",
)

# Set up CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For development; restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(proposal_router)


@app.on_event("startup")
async def startup_event():
    """Initialize components on startup"""
    # Check for database connection
    db_connection = os.environ.get("DB_CONNECTION")
    if db_connection:
        logger.info("Database connection found in environment variables")
    else:
        logger.warning("DB_CONNECTION not found in environment variables")
        logger.warning(
            "Using default connection string or fallback to in-memory storage"
        )

    # Initialize database connection and tables
    await SessionManager.initialize()
    logger.info("Session storage initialized")

    # Configure LlamaIndex with Gemini
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        logger.warning("GOOGLE_API_KEY not found in environment variables")
        logger.warning("The application will not be able to use Gemini LLM")

    # Initialize LLM config
    llm_config = LLMConfig(api_key=api_key)
    settings = llm_config.setup_llama_index()

    logger.info("Proposal Generator API started")

    # Log available models if API key is set
    if api_key:
        models = llm_config.list_available_models()
        if models:
            logger.info(f"Available Gemini models: {', '.join(models)}")


@app.get("/")
async def root():
    """Root endpoint - API information"""
    return {
        "name": "Proposal Generator API",
        "version": "0.1.0",
        "description": "Generate customized proposals using LlamaIndex and Gemini",
        "endpoints": "/docs or /redoc for API documentation",
    }


if __name__ == "__main__":
    # Run the application
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
