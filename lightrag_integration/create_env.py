#!/usr/bin/env python3
"""
Create LightRAG .env file with persistent storage configuration
"""

import os
from pathlib import Path

# Base project directory
PROJECT_DIR = Path("/Users/aniket.rastogi/Documents/patent_project")

# LightRAG Environment Variables for PERSISTENT STORAGE
LIGHTRAG_ENV_VARS = {
    # Server Configuration
    "HOST": "0.0.0.0",
    "PORT": "9621",
    "WORKERS": "2",
    "MAX_ASYNC": "4",
    "TIMEOUT": "200",
    "TEMPERATURE": "0.0",
    "MAX_TOKENS": "8192",
    
    # LLM Configuration - Ollama with Qwen2.5:14b-instruct
    "LLM_BINDING": "ollama",
    "LLM_MODEL": "qwen2.5:14b-instruct",
    "LLM_BINDING_HOST": "http://localhost:11434",
    
    # Embedding Configuration - Ollama with BGE-M3
    "EMBEDDING_BINDING": "ollama",
    "EMBEDDING_MODEL": "bge-m3:latest",
    "EMBEDDING_BINDING_HOST": "http://localhost:11434",
    "EMBEDDING_DIM": "1024",
    
    # PERSISTENT STORAGE Configuration
    "LIGHTRAG_GRAPH_STORAGE": "Neo4JStorage",
    "LIGHTRAG_KV_STORAGE": "JsonKVStorage",
    "LIGHTRAG_VECTOR_STORAGE": "NanoVectorDBStorage", 
    "LIGHTRAG_DOC_STATUS_STORAGE": "JsonDocStatusStorage",
    
    # Neo4j Configuration for PERSISTENCE
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
    
    # PERSISTENCE PATHS
    "LIGHTRAG_WORKING_DIR": str(PROJECT_DIR / "lightrag_storage"),
    "LIGHTRAG_INPUT_DIR": str(PROJECT_DIR / "hupd_processed"),
    "LIGHTRAG_DATA_DIR": str(PROJECT_DIR / "lightrag_data"),
}

def create_env_file():
    """Create .env file for LightRAG with persistent storage"""
    env_file_path = PROJECT_DIR / ".env"
    
    with open(env_file_path, 'w') as f:
        for key, value in LIGHTRAG_ENV_VARS.items():
            f.write(f"{key}={value}\n")
    
    print(f"✅ Created .env file at {env_file_path}")
    print("📁 Persistent storage configured:")
    print(f"   - Working Directory: {LIGHTRAG_ENV_VARS['LIGHTRAG_WORKING_DIR']}")
    print(f"   - Neo4j Graph Storage: {LIGHTRAG_ENV_VARS['NEO4J_URI']}")
    print(f"   - JSON KV Storage: {LIGHTRAG_ENV_VARS['LIGHTRAG_KV_STORAGE']}")
    print(f"   - Vector Storage: {LIGHTRAG_ENV_VARS['LIGHTRAG_VECTOR_STORAGE']}")
    
    return env_file_path

if __name__ == "__main__":
    create_env_file() 