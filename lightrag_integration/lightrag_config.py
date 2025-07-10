#!/usr/bin/env python3
"""
LightRAG Configuration for Patent Analysis System
Integration with Ollama and Qwen2.5:14b-instruct model
"""

import os
from pathlib import Path

# Base project directory
PROJECT_DIR = Path("/Users/aniket.rastogi/Documents/patent_project")

# LightRAG Server Configuration
LIGHTRAG_HOST = "0.0.0.0"
LIGHTRAG_PORT = 9621
LIGHTRAG_BASE_URL = f"http://{LIGHTRAG_HOST}:{LIGHTRAG_PORT}"

# Ollama Configuration
OLLAMA_HOST = "http://localhost:11434"
OLLAMA_MODEL = "qwen2.5:14b-instruct"
OLLAMA_EMBEDDING_MODEL = "bge-m3:latest"

# LightRAG Environment Variables
LIGHTRAG_ENV_VARS = {
    # Server Configuration
    "HOST": LIGHTRAG_HOST,
    "PORT": str(LIGHTRAG_PORT),
    "WORKERS": "2",
    "MAX_ASYNC": "4",
    "TIMEOUT": "200",
    "TEMPERATURE": "0.0",
    "MAX_TOKENS": "8192",
    
    # LLM Configuration - Ollama with Qwen2.5:14b-instruct
    "LLM_BINDING": "ollama",
    "LLM_MODEL": OLLAMA_MODEL,
    "LLM_BINDING_HOST": OLLAMA_HOST,
    
    # Embedding Configuration - Ollama with BGE-M3
    "EMBEDDING_BINDING": "ollama",
    "EMBEDDING_MODEL": OLLAMA_EMBEDDING_MODEL,
    "EMBEDDING_BINDING_HOST": OLLAMA_HOST,
    "EMBEDDING_DIM": "1024",
    
    # Storage Configuration - Using Neo4j for graph storage
    "LIGHTRAG_GRAPH_STORAGE": "Neo4JStorage",
    "LIGHTRAG_KV_STORAGE": "JsonKVStorage",
    "LIGHTRAG_VECTOR_STORAGE": "NanoVectorDBStorage",
    "LIGHTRAG_DOC_STATUS_STORAGE": "JsonDocStatusStorage",
    
    # Neo4j Configuration
    "NEO4J_URI": "bolt://localhost:7687",
    "NEO4J_USERNAME": "neo4j",
    "NEO4J_PASSWORD": "password",
    
    # Document Processing
    "ENABLE_LLM_CACHE_FOR_EXTRACT": "true",
    "SUMMARY_LANGUAGE": "English",
    "MAX_PARALLEL_INSERT": "1",
    
    # Query Configuration
    "TOP_K": "50",
    "COSINE_THRESHOLD": "0.4",
}

# LightRAG Directories
LIGHTRAG_WORKING_DIR = PROJECT_DIR / "lightrag_storage"
LIGHTRAG_INPUT_DIR = PROJECT_DIR / "hupd_processed"
LIGHTRAG_DATA_DIR = PROJECT_DIR / "lightrag_data"

# Create directories if they don't exist
LIGHTRAG_WORKING_DIR.mkdir(exist_ok=True)
LIGHTRAG_INPUT_DIR.mkdir(exist_ok=True)
LIGHTRAG_DATA_DIR.mkdir(exist_ok=True)

def create_env_file():
    """Create .env file for LightRAG"""
    env_file_path = PROJECT_DIR / ".env"
    
    with open(env_file_path, 'w') as f:
        for key, value in LIGHTRAG_ENV_VARS.items():
            f.write(f"{key}={value}\n")
    
    print(f"Created .env file at {env_file_path}")
    return env_file_path

def get_lightrag_api_url(endpoint=""):
    """Get LightRAG API URL for a specific endpoint"""
    return f"{LIGHTRAG_BASE_URL}{endpoint}"

def validate_ollama_connection():
    """Validate that Ollama is running and Qwen2.5:14b-instruct is available"""
    import requests
    
    try:
        # Check if Ollama is running
        response = requests.get(f"{OLLAMA_HOST}/api/tags", timeout=5)
        if response.status_code == 200:
            models = response.json().get('models', [])
            qwen_model = next((m for m in models if m.get('name') == OLLAMA_MODEL), None)
            
            if qwen_model:
                print(f"✅ Ollama is running and {OLLAMA_MODEL} is available")
                return True
            else:
                print(f"⚠️  Ollama is running but {OLLAMA_MODEL} is not found")
                print("Available models:", [m.get('name') for m in models])
                return False
        else:
            print(f"❌ Ollama is not responding (status: {response.status_code})")
            return False
    except requests.exceptions.RequestException as e:
        print(f"❌ Cannot connect to Ollama: {e}")
        return False

def validate_lightrag_connection():
    """Validate that LightRAG server is running"""
    import requests
    
    try:
        response = requests.get(f"{LIGHTRAG_BASE_URL}/health", timeout=5)
        if response.status_code == 200:
            print("✅ LightRAG server is running")
            return True
        else:
            print(f"❌ LightRAG server is not responding (status: {response.status_code})")
            return False
    except requests.exceptions.RequestException as e:
        print(f"❌ Cannot connect to LightRAG server: {e}")
        return False 