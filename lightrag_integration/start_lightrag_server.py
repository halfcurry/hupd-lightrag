#!/usr/bin/env python3
"""
Script to start LightRAG server using its main() entrypoint with persistent storage.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

def start_server():
    # Load environment variables from .env file
    project_dir = Path("/Users/aniket.rastogi/Documents/patent_project")
    env_file = project_dir / ".env"
    
    if env_file.exists():
        load_dotenv(env_file)
        print(f"✅ Loaded environment from {env_file}")
    else:
        print(f"⚠️  No .env file found at {env_file}")
    
    # Set working directory for LightRAG
    working_dir = project_dir / "rag_storage"
    working_dir.mkdir(exist_ok=True)
    os.chdir(working_dir)
    
    print(f"📁 LightRAG working directory: {working_dir}")
    print("🔧 Starting LightRAG server with persistent storage...")
    
    from lightrag.api.lightrag_server import main
    main()

if __name__ == "__main__":
    start_server() 