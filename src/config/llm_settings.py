"""
LLM and Embedding Model Configuration
====================================
Configuration for RDL migration tool.
Replace the functions below with your organization's implementation.
"""

import os
import logging
from typing import Optional, Tuple

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# =============================================================================
# FUNCTIONS TO REPLACE FOR ORGANIZATION DEPLOYMENT
# =============================================================================
# Replace these 2 functions with your organization's Azure implementation:

def get_llm():
    """
    *** REPLACE THIS FUNCTION FOR ORGANIZATION ***
    
    Current: OpenAI implementation
    Organization: Replace with your Azure ChatOpenAI implementation
    
    Must return: LangChain chat model or None
    """
    try:
        logger.info("Initializing OpenAI LLM client")
        from langchain_openai import ChatOpenAI
        
        current_api_key = os.getenv("OPENAI_API_KEY")
        if not current_api_key:
            logger.error("OPENAI_API_KEY environment variable not set")
            return None
        
        return ChatOpenAI(
            model="gpt-4o",
            temperature=0.5,
            max_tokens=16000
        )
        
    except Exception as e:
        logger.error(f"Error creating LLM client: {e}")
        return None

def get_embedding_model():
    """
    *** REPLACE THIS FUNCTION FOR ORGANIZATION ***
    
    Current: OpenAI implementation  
    Organization: Replace with your Azure OpenAIEmbeddings implementation
    
    Must return: LangChain embeddings model or None
    """
    try:
        logger.info("Initializing OpenAI embedding model")
        from langchain_openai import OpenAIEmbeddings
        
        current_api_key = os.getenv("OPENAI_API_KEY")
        if not current_api_key:
            logger.error("OPENAI_API_KEY environment variable not set")
            return None
        
        return OpenAIEmbeddings(
            model="text-embedding-3-large"
        )
        
    except Exception as e:
        logger.error(f"Error creating embedding model: {e}")
        return None

# =============================================================================
# COMMON CODE - DO NOT CHANGE FOR ORGANIZATION
# =============================================================================
# This code works with both OpenAI and Azure implementations:

def refresh_clients() -> Tuple[Optional[object], Optional[object]]:
    """
    Force refresh of both LLM and embedding clients.
    This function works with both OpenAI and Azure implementations.
    """
    logger.info("Refreshing LLM and embedding clients...")
    global llm, embedding_model
    llm = get_llm()
    embedding_model = get_embedding_model()
    return llm, embedding_model

# Initialize global clients with lazy loading
def _initialize_clients():
    """Initialize clients with lazy loading to handle Flask reloader issues"""
    global llm, embedding_model
    
    if 'llm' not in globals() or llm is None:
        logger.info("Initializing global LLM and embedding clients...")
        llm = get_llm()
        embedding_model = get_embedding_model()
        
        # Verify initialization
        if llm:
            logger.info("SUCCESS: LLM client initialized successfully")
        else:
            logger.error("ERROR: Failed to initialize LLM client")
            
        if embedding_model:
            logger.info("SUCCESS: Embedding model initialized successfully")
        else:
            logger.error("ERROR: Failed to initialize embedding model")

# Try initial initialization
llm = None
embedding_model = None
_initialize_clients()

# Provide getter functions for Flask reloader compatibility
def get_llm_client():
    """Get LLM client with lazy initialization"""
    global llm
    if llm is None:
        llm = get_llm()
    return llm

def get_embedding_client():
    """Get embedding client with lazy initialization"""  
    global embedding_model
    if embedding_model is None:
        embedding_model = get_embedding_model()
    return embedding_model

# Export for use in other modules
__all__ = [
    'get_llm', 
    'get_embedding_model',
    'refresh_clients',
    'get_llm_client',
    'get_embedding_client', 
    'llm',
    'embedding_model'
]