#!/usr/bin/env python3
"""
Main Patent Analysis Pipeline with Persistent Storage
Orchestrates the complete LightRAG pipeline from filtering to chatbot
Ensures data persistence across server restarts
"""

import argparse
import logging
import signal
import sys
import time
import subprocess
import requests
from pathlib import Path
from typing import Dict, List, Optional
import json
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration
LIGHTRAG_BASE_URL = "http://localhost:9621"
OLLAMA_HOST = "http://localhost:11434"

class PatentAnalysisPipeline:
    def __init__(self):
        self.running = True
        self.lightrag_process = None
        
        # Set up signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        logger.info("Received shutdown signal, stopping gracefully...")
        self.running = False
        if self.lightrag_process:
            self.lightrag_process.terminate()
    
    def check_dependencies(self) -> bool:
        """Check if all required dependencies are available"""
        logger.info("Checking dependencies...")
        
        # Check if virtual environment is activated
        if not hasattr(sys, 'real_prefix') and not (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
            logger.warning("Virtual environment not detected. Please activate your virtual environment.")
        
        # Check required directories
        required_dirs = ['hupd_extracted', 'hupd_processed', 'hupd_optimized', 'lightrag_upload', 'lightrag_storage']
        for dir_name in required_dirs:
            Path(dir_name).mkdir(exist_ok=True)
            logger.info(f"✅ Directory ready: {dir_name}")
        
        # Check if Ollama is running
        try:
            response = requests.get(f"{OLLAMA_HOST}/api/tags", timeout=5)
            if response.status_code == 200:
                models = response.json().get('models', [])
                qwen_model = next((m for m in models if m.get('name') == "qwen2.5:14b-instruct"), None)
                if qwen_model:
                    logger.info("✅ Ollama is running with qwen2.5:14b-instruct model")
                else:
                    logger.error("❌ qwen2.5:14b-instruct model not found in Ollama")
                    return False
            else:
                logger.error("❌ Ollama is not responding")
                return False
        except Exception as e:
            logger.error(f"❌ Cannot connect to Ollama: {e}")
            return False
        
        # Check if LightRAG CLI is available (optional)
        try:
            result = subprocess.run(["lightrag", "--version"], capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                logger.info("✅ LightRAG CLI is available")
            else:
                logger.warning("⚠️ LightRAG CLI not found - you may need to start LightRAG server manually")
        except (FileNotFoundError, subprocess.TimeoutExpired):
            logger.warning("⚠️ LightRAG CLI not found - you may need to start LightRAG server manually")
        except Exception as e:
            logger.warning(f"⚠️ Could not check LightRAG CLI: {e}")
        
        logger.info("✅ All dependencies are ready")
        return True
    
    def filter_patents(self, input_dir: str, output_dir: str = "hupd_processed") -> bool:
        """Filter G06 patents and optimize them"""
        logger.info("Starting patent filtering and optimization...")
        
        try:
            # Run the filtering script
            result = subprocess.run([
                sys.executable, 
                "filtering/filter_g06_patents_optimized.py",
                input_dir,
                "--output", output_dir
            ], capture_output=True, text=True)
            
            if result.returncode == 0:
                logger.info("✅ Patent filtering completed successfully")
                return True
            else:
                logger.error(f"❌ Patent filtering failed: {result.stderr}")
                return False
        except Exception as e:
            logger.error(f"❌ Error during patent filtering: {e}")
            return False
    
    def start_lightrag_server(self) -> bool:
        """Start the LightRAG server with persistent storage"""
        logger.info("Starting LightRAG server with persistent storage...")
        
        try:
            # Create .env file for LightRAG with persistent storage
            from lightrag_integration.create_env import create_env_file
            env_file = create_env_file()
            logger.info(f"✅ Created persistent storage configuration: {env_file}")
            
            # Try to start LightRAG server using the updated start script
            try:
                # Use the updated start script that loads .env file
                self.lightrag_process = subprocess.Popen([
                    sys.executable, 
                    "lightrag_integration/start_lightrag_server.py"
                ], cwd=".")
                
                # Wait for server to start
                logger.info("Waiting for LightRAG server to start...")
                for i in range(30):  # Wait up to 30 seconds
                    try:
                        response = requests.get(f"{LIGHTRAG_BASE_URL}/health", timeout=5)
                        if response.status_code == 200:
                            logger.info("✅ LightRAG server is running with persistent storage")
                            return True
                    except:
                        pass
                    time.sleep(1)
                
                logger.error("❌ LightRAG server failed to start")
                return False
                
            except FileNotFoundError:
                logger.warning("⚠️ LightRAG server script not found. Please start manually:")
                logger.info(f"   python lightrag_integration/start_lightrag_server.py")
                logger.info("   Or use the web interface directly if LightRAG is already running.")
                return False
            
        except Exception as e:
            logger.error(f"❌ Error starting LightRAG server: {e}")
            return False
    
    def check_lightrag_health(self) -> bool:
        """Check if LightRAG server is healthy"""
        try:
            response = requests.get(f"{LIGHTRAG_BASE_URL}/health", timeout=10)
            if response.status_code == 200:
                data = response.json()
                if data.get("status") == "healthy":
                    return True
            return False
        except:
            return False
    
    def integrate_patents(self, input_dir: str = "hupd_processed") -> bool:
        """Integrate patents into LightRAG"""
        logger.info("Starting patent integration into LightRAG...")
        
        try:
            # Run the integration script
            result = subprocess.run([
                sys.executable,
                "lightrag_integration/integrate_lightrag_g06_patents_sequential.py"
            ], capture_output=True, text=True)
            
            if result.returncode == 0:
                logger.info("✅ Patent integration completed successfully")
                return True
            else:
                logger.error(f"❌ Patent integration failed: {result.stderr}")
                return False
        except Exception as e:
            logger.error(f"❌ Error during patent integration: {e}")
            return False
    
    def test_chatbot(self) -> bool:
        """Test the LightRAG chatbot"""
        logger.info("Testing LightRAG chatbot...")
        
        try:
            # Test query
            test_query = {
                "model": "qwen2.5:14b-instruct",
                "messages": [
                    {
                        "role": "user",
                        "content": "What are some recent G06 patents related to computer technology?"
                    }
                ],
                "stream": True
            }
            
            response = requests.post(
                f"{LIGHTRAG_BASE_URL}/api/chat",
                json=test_query,
                timeout=60,
                stream=True
            )
            
            if response.status_code == 200:
                logger.info("✅ Chatbot test successful - received streaming response")
                return True
            else:
                logger.error(f"❌ Chatbot test failed: {response.status_code}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Error testing chatbot: {e}")
            return False
    
    def launch_chatbot(self) -> bool:
        """Launch the interactive chatbot interface"""
        logger.info("Launching interactive chatbot interface...")
        
        try:
            # Import and run the chatbot
            from chatbot.patent_chatbot import main as chatbot_main
            chatbot_main()
            return True
        except Exception as e:
            logger.error(f"❌ Error launching chatbot: {e}")
            return False
    
    def show_pipeline_status(self):
        """Show the current status of the pipeline with persistent storage info"""
        logger.info("\n" + "="*60)
        logger.info("PIPELINE STATUS (with Persistent Storage)")
        logger.info("="*60)
        
        # Check directories
        directories = {
            "Source Data": Path("hupd_extracted"),
            "Filtered Patents": Path("hupd_processed"),
            "LightRAG Upload": Path("lightrag_upload"),
            "LightRAG Storage": Path("lightrag_storage")
        }
        
        for name, path in directories.items():
            if path.exists():
                file_count = len(list(path.glob("*.json")))
                size_mb = sum(f.stat().st_size for f in path.glob("*.json")) / (1024*1024)
                logger.info(f"✅ {name}: {file_count} files ({size_mb:.1f} MB)")
            else:
                logger.info(f"❌ {name}: Directory not found")
        
        # Check persistent storage configuration
        env_file = Path(".env")
        if env_file.exists():
            logger.info(f"✅ Persistent Storage Config: {env_file}")
        else:
            logger.info(f"❌ Persistent Storage Config: Missing .env file")
        
        # Check Neo4j for graph storage
        try:
            import subprocess
            result = subprocess.run([
                "cypher-shell", "-u", "neo4j", "-p", "password", 
                "MATCH (n) RETURN count(n) as total_nodes"
            ], capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')
                if len(lines) >= 3:
                    node_count = lines[2].strip()
                    logger.info(f"📊 Neo4j Graph Storage: {node_count} nodes")
                else:
                    logger.info(f"❌ Neo4j Graph Storage: No data")
            else:
                logger.info(f"❌ Neo4j Graph Storage: Connection failed")
        except:
            logger.info(f"❌ Neo4j Graph Storage: Not available")
        
        # Check services
        services = {
            "Ollama": f"{OLLAMA_HOST}/api/tags",
            "LightRAG": f"{LIGHTRAG_BASE_URL}/health"
        }
        
        for name, url in services.items():
            try:
                response = requests.get(url, timeout=5)
                if response.status_code == 200:
                    logger.info(f"✅ {name}: Running")
                else:
                    logger.info(f"❌ {name}: Not responding")
            except:
                logger.info(f"❌ {name}: Not available")
        
        logger.info("="*60)
    
    def run_full_pipeline(self, input_dir: str, skip_filtering: bool = False) -> bool:
        """Run the complete pipeline"""
        logger.info("Starting complete patent analysis pipeline...")
        
        # Step 1: Check dependencies
        if not self.check_dependencies():
            logger.error("❌ Dependency check failed")
            return False
        
        # Step 2: Filter patents (if not skipped)
        if not skip_filtering:
            if not self.filter_patents(input_dir):
                logger.error("❌ Patent filtering failed")
                return False
        else:
            logger.info("⏭️ Skipping patent filtering")
        
        # Step 3: Start LightRAG server
        if not self.start_lightrag_server():
            logger.error("❌ LightRAG server failed to start")
            return False
        
        # Step 4: Integrate patents
        if not self.integrate_patents():
            logger.error("❌ Patent integration failed")
            return False
        
        # Step 5: Test chatbot
        if not self.test_chatbot():
            logger.error("❌ Chatbot test failed")
            return False
        
        # Step 6: Show final status
        self.show_pipeline_status()
        
        logger.info("🎉 Pipeline completed successfully!")
        logger.info("You can now access the LightRAG chatbot at: http://localhost:9621")
        
        return True
    
    def run_interactive_mode(self):
        """Run in interactive mode with menu"""
        while self.running:
            print("\n" + "="*50)
            print("PATENT ANALYSIS PIPELINE")
            print("="*50)
            print("1. Check dependencies")
            print("2. Filter patents")
            print("3. Start LightRAG server")
            print("4. Integrate patents")
            print("5. Test chatbot")
            print("6. Launch chatbot interface")
            print("7. Show status")
            print("8. Run full pipeline")
            print("9. Exit")
            print("="*50)
            
            choice = input("Select an option (1-9): ").strip()
            
            if choice == "1":
                self.check_dependencies()
            elif choice == "2":
                input_dir = input("Enter input directory (default: hupd_extracted): ").strip() or "hupd_extracted"
                self.filter_patents(input_dir)
            elif choice == "3":
                self.start_lightrag_server()
            elif choice == "4":
                self.integrate_patents()
            elif choice == "5":
                self.test_chatbot()
            elif choice == "6":
                self.launch_chatbot()
            elif choice == "7":
                self.show_pipeline_status()
            elif choice == "8":
                input_dir = input("Enter input directory (default: hupd_extracted): ").strip() or "hupd_extracted"
                skip_filtering = input("Skip filtering? (y/N): ").strip().lower() == 'y'
                self.run_full_pipeline(input_dir, skip_filtering)
            elif choice == "9":
                logger.info("Exiting...")
                break
            else:
                print("Invalid choice. Please try again.")

def main():
    parser = argparse.ArgumentParser(description="Patent Analysis Pipeline with Persistent Storage")
    parser.add_argument("--mode", choices=["full", "interactive", "check", "filter", "start-server", "integrate", "test", "chatbot"], 
                       default="interactive", help="Operation mode")
    parser.add_argument("--input-dir", default="hupd_extracted", help="Input directory for patent data")
    parser.add_argument("--skip-filtering", action="store_true", help="Skip patent filtering step")
    
    args = parser.parse_args()
    
    pipeline = PatentAnalysisPipeline()
    
    try:
        if args.mode == "full":
            success = pipeline.run_full_pipeline(args.input_dir, args.skip_filtering)
            return 0 if success else 1
        elif args.mode == "interactive":
            pipeline.run_interactive_mode()
            return 0
        elif args.mode == "check":
            success = pipeline.check_dependencies()
            return 0 if success else 1
        elif args.mode == "filter":
            success = pipeline.filter_patents(args.input_dir)
            return 0 if success else 1
        elif args.mode == "start-server":
            success = pipeline.start_lightrag_server()
            return 0 if success else 1
        elif args.mode == "integrate":
            success = pipeline.integrate_patents()
            return 0 if success else 1
        elif args.mode == "test":
            success = pipeline.test_chatbot()
            return 0 if success else 1
        elif args.mode == "chatbot":
            success = pipeline.launch_chatbot()
            return 0 if success else 1
    
    except KeyboardInterrupt:
        logger.info("Operation cancelled by user")
        return 0
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return 1

if __name__ == "__main__":
    exit(main()) 