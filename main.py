#!/usr/bin/env python3
"""
Patent Analysis Pipeline Orchestrator

This script provides a unified interface for:
1. Filtering patents by IPC labels
2. Uploading documents to LightRAG
3. Running the chatbot with guardrails
4. Evaluating responses with comprehensive metrics
"""

import os
import sys
import subprocess
import time
import logging
from typing import List, Dict
import argparse
from datetime import datetime

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import filtering module
from filtering.filter_g06_patents_optimized import OptimizedG06PatentFilter

# Import LightRAG integration
from lightrag_integration.lightrag_uploader import LightRAGUploader

# Import chatbot
from chatbot.patent_chatbot import PatentChatbot

# Import evaluation module
from evaluation.evaluate_responses import ResponseEvaluator

# Import backup management
from backup.manage_backup import BackupManager

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PatentPipelineOrchestrator:
    """
    Orchestrates the complete patent analysis pipeline
    """
    
    def __init__(self):
        self.filter = OptimizedG06PatentFilter  # Use the class directly
        self.uploader = LightRAGUploader()
        self.chatbot = None  # Will be initialized with guardrails flag as needed
        self.evaluator = None  # Will be initialized when needed
        self.monitor = None  # Will be initialized when needed
        
    def run_filtering(self, source_dir: str = "hupd_processed", output_dir: str = "hupd_processed"):
        """Run patent filtering by IPC labels"""
        print("🔍 Starting patent filtering...")
        try:
            filter_instance = self.filter(source_dir, output_dir)
            filter_instance.process_directory()
            print("✅ Patent filtering completed successfully!")
        except Exception as e:
            print(f"❌ Error during filtering: {e}")
    
    def run_upload(self, source_dir: str = "hupd_processed"):
        """Run document upload to LightRAG"""
        print("📤 Starting document upload to LightRAG...")
        try:
            self.uploader.upload_documents_sequential(source_dir)
            print("✅ Document upload completed successfully!")
        except Exception as e:
            print(f"❌ Error during upload: {e}")
    
    def run_chatbot(self, with_guardrails: bool = True, enable_monitoring: bool = True, web_interface: bool = False):
        """Run the interactive chatbot with or without guardrails and monitoring"""
        print(f"🤖 Starting Patent Analysis Assistant...")
        print(f"   Interface: {'🌐 Web (Gradio)' if web_interface else '💻 CLI'}")
        print(f"   Guardrails: {'ENABLED' if with_guardrails else 'DISABLED'}")
        print(f"   Monitoring: {'ENABLED' if enable_monitoring else 'DISABLED'}")
        
        try:
            self.chatbot = PatentChatbot(
                with_guardrails=with_guardrails,
                enable_monitoring=enable_monitoring
            )
            
            if web_interface:
                self.chatbot.run_gradio_interface()
            else:
                self.chatbot.interactive_chat()
                
        except Exception as e:
            print(f"❌ Error running chatbot: {e}")
        finally:
            if self.chatbot:
                self.chatbot.cleanup()
    
    def run_evaluation(self, queries: List[str] = None):
        """Run comprehensive evaluation with guardrails metrics"""
        print("📊 Starting comprehensive evaluation...")
        
        # Initialize evaluator only when needed
        if self.evaluator is None:
            print("🔄 Initializing evaluation models...")
            self.evaluator = ResponseEvaluator()
        
        if queries is None:
            queries = [
                "What is the main claim of this patent?",
                "How does this invention work?",
                "What are the key features of this patent?",
                "What is the technical background of this invention?",
                "How does this patent compare to prior art?",
                "What are the advantages of this invention?",
                "How is this patent classified?",
                "What is the scope of protection?",
                "What are the potential applications?",
                "How does this relate to existing technology?"
            ]
        
        try:
            # Run batch evaluation
            summary = self.chatbot.batch_evaluate(queries)
            self.chatbot.print_batch_report(summary)
            
            # Save results to file
            self._save_evaluation_results(summary, queries)
            
            print("✅ Evaluation completed successfully!")
            
        except Exception as e:
            print(f"❌ Error during evaluation: {e}")
    
    def _save_evaluation_results(self, summary: Dict, queries: List[str]):
        """Save evaluation results to a JSON file"""
        try:
            results = {
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "queries": queries,
                "summary": summary
            }
            
            filename = f"evaluation_results_{int(time.time())}.json"
            with open(filename, 'w') as f:
                import json
                json.dump(results, f, indent=2, default=str)
            
            print(f"📄 Results saved to: {filename}")
            
        except Exception as e:
            print(f"⚠️  Could not save results: {e}")
    
    def run_full_pipeline(self):
        """Run the complete pipeline from filtering to evaluation"""
        print("🚀 Starting complete patent analysis pipeline...")
        print("=" * 60)
        
        try:
            # Step 1: Filter patents
            print("\n1️⃣  Step 1: Filtering patents by IPC labels...")
            self.run_filtering()
            
            # Step 2: Upload to LightRAG
            print("\n2️⃣  Step 2: Uploading documents to LightRAG...")
            self.run_upload()
            
            # Step 3: Run evaluation
            print("\n3️⃣  Step 3: Running comprehensive evaluation...")
            self.run_evaluation()
            
            print("\n✅ Complete pipeline finished successfully!")
            
        except Exception as e:
            print(f"❌ Pipeline failed: {e}")
    
    def check_lightrag_status(self) -> bool:
        """Check if LightRAG server is running"""
        try:
            import requests
            response = requests.get("http://localhost:9621/health", timeout=5)
            return response.status_code == 200
        except:
            return False
    
    def show_monitoring_dashboard(self):
        """Show real-time monitoring dashboard"""
        print("\n📊 MONITORING DASHBOARD")
        print("=" * 50)
        
        if not self.chatbot:
            print("❌ No chatbot instance available. Start the chatbot first.")
            return
        
        if hasattr(self.chatbot, 'integrated_monitor') and self.chatbot.integrated_monitor:
            # Use integrated monitoring dashboard
            self.chatbot.integrated_monitor.display_monitoring_dashboard()
        elif self.chatbot.monitor:
            # Use legacy monitoring
            try:
                summary = self.chatbot.get_monitoring_summary()
                
                # Performance metrics
                if 'performance' in summary and 'error' not in summary['performance']:
                    perf = summary['performance']
                    print(f"\n📈 PERFORMANCE (Last Hour):")
                    print(f"   Total Requests: {perf.get('total_requests', 0)}")
                    print(f"   Success Rate: {perf.get('success_rate', 0):.1%}")
                    print(f"   Avg Response Time: {perf.get('avg_response_time', 0):.2f}s")
                    print(f"   P95 Response Time: {perf.get('p95_response_time', 0):.2f}s")
                    print(f"   Error Count: {perf.get('error_count', 0)}")
                
                # System health
                if 'system_health' in summary and 'error' not in summary['system_health']:
                    health = summary['system_health']
                    current = health.get('current_status', {})
                    print(f"\n🛡️  SYSTEM HEALTH:")
                    print(f"   LightRAG: {'🟢 UP' if current.get('lightrag') else '🔴 DOWN'}")
                    print(f"   Neo4j: {'🟢 UP' if current.get('neo4j') else '🔴 DOWN'}")
                    print(f"   Ollama: {'🟢 UP' if current.get('ollama') else '🔴 DOWN'}")
                    print(f"   CPU Usage: {current.get('cpu_usage', 0):.1f}%")
                    print(f"   Memory Usage: {current.get('memory_usage', 0):.1f}%")
                    print(f"   Active Sessions: {current.get('active_connections', 0)}")
                
                # Real-time metrics
                if 'grafana_export' in summary:
                    realtime = summary['grafana_export'].get('real_time', {})
                    print(f"\n⏱️  REAL-TIME METRICS:")
                    print(f"   Active Sessions: {realtime.get('active_sessions', 0)}")
                    print(f"   Total Requests: {realtime.get('total_requests', 0)}")
                    print(f"   Error Count: {realtime.get('error_count', 0)}")
                    print(f"   Avg Response Time: {realtime.get('avg_response_time', 0):.2f}s")
                
            except Exception as e:
                print(f"❌ Error getting monitoring data: {e}")
        else:
            print("❌ No monitoring data available. Start the chatbot with monitoring enabled.")
    
    def save_monitoring_data(self):
        """Save monitoring data to file"""
        if not self.chatbot:
            print("❌ No chatbot instance available. Start the chatbot first.")
            return
        
        try:
            if hasattr(self.chatbot, 'integrated_monitor') and self.chatbot.integrated_monitor:
                # Use integrated monitoring
                filename = self.chatbot.integrated_monitor.save_monitoring_data()
                print(f"✅ Integrated monitoring data saved to: {filename}")
            elif self.chatbot.monitor:
                # Use legacy monitoring
                filename = self.chatbot.save_monitoring_data()
                print(f"✅ Monitoring data saved to: {filename}")
            else:
                print("❌ No monitoring data available. Start the chatbot with monitoring enabled.")
        except Exception as e:
            print(f"❌ Error saving monitoring data: {e}")
    
    def interactive_menu(self):
        """Display interactive menu"""
        while True:
            print("\n" + "="*60)
            print("🔬 PATENT ANALYSIS PIPELINE")
            print("="*60)
            print("1. 🔍 Filter patents by IPC labels")
            print("2. 📤 Upload documents to LightRAG")
            print("3. 🤖 Run interactive chatbot (CLI)")
            print("4. 🌐 Run interactive chatbot (Web)")
            print("5. 📊 Run comprehensive evaluation")
            print("6. 🚀 Run complete pipeline")
            print("7. 🔍 Check LightRAG status")
            print("8. 📋 Show evaluation metrics")
            print("9. 🛡️  Test guardrails validation")
            print("10. 📈 Show monitoring dashboard")
            print("11. 💾 Save monitoring data")
            print("12. 💾 Test SQLite fallback")
            print("13. ❌ Exit")
            print("="*60)
            
            choice = input("Select an option (1-13): ").strip()
            
            if choice == "1":
                self.run_filtering()
            elif choice == "2":
                self.run_upload()
            elif choice == "3":
                # Prompt for guardrails and monitoring
                g = input("Enable guardrails? (Y/n): ").strip().lower()
                with_guardrails = not (g == 'n')
                
                m = input("Enable monitoring? (Y/n): ").strip().lower()
                enable_monitoring = not (m == 'n')
                
                self.run_chatbot(with_guardrails=with_guardrails, enable_monitoring=enable_monitoring, web_interface=False)
            elif choice == "4":
                # Prompt for guardrails and monitoring
                g = input("Enable guardrails? (Y/n): ").strip().lower()
                with_guardrails = not (g == 'n')
                
                m = input("Enable monitoring? (Y/n): ").strip().lower()
                enable_monitoring = not (m == 'n')
                
                self.run_chatbot(with_guardrails=with_guardrails, enable_monitoring=enable_monitoring, web_interface=True)
            elif choice == "5":
                self.run_evaluation()
            elif choice == "6":
                self.run_full_pipeline()
            elif choice == "7":
                status = "🟢 Running" if self.check_lightrag_status() else "🔴 Not running"
                print(f"LightRAG Status: {status}")
            elif choice == "8":
                self._show_evaluation_metrics()
            elif choice == "9":
                self._test_guardrails()
            elif choice == "10":
                self.show_monitoring_dashboard()
            elif choice == "11":
                self.save_monitoring_data()
            elif choice == "12":
                self._test_sqlite_fallback()
            elif choice == "13":
                print("👋 Goodbye!")
                break
            else:
                print("❌ Invalid option. Please try again.")
    
    def _show_evaluation_metrics(self):
        """Show available evaluation metrics"""
        print("\n📊 EVALUATION METRICS")
        print("=" * 40)
        print("Traditional Metrics:")
        print("• ROUGE-1: N-gram overlap precision")
        print("• ROUGE-2: Bigram overlap precision")
        print("• ROUGE-L: Longest common subsequence")
        print("• Relevance: Semantic similarity to query")
        print("• Coherence: Text quality and structure")
        
        print("\n🛡️  Guardrails Metrics:")
        print("• Profanity Score: Content appropriateness")
        print("• Topic Relevance: Patent-related content")
        print("• Politeness Score: Professional tone")
        
        print("\n📈 Overall Scores:")
        print("• Combined evaluation score")
        print("• Guardrails compliance score")
        print("• Response time and source count")
    
    def _test_guardrails(self):
        """Test guardrails validation with sample responses"""
        print("\n🛡️  TESTING GUARDRAILS VALIDATION")
        print("=" * 50)
        
        test_responses = [
            "This patent describes a novel method for data encryption.",
            "This is a terrible patent that should be rejected immediately!",
            "The invention relates to a new type of semiconductor device.",
            "I don't care about your stupid patent questions."
        ]
        
        from chatbot.guardrails_validator import GuardrailsValidator
        validator = GuardrailsValidator()
        
        for i, response in enumerate(test_responses, 1):
            print(f"\nTest {i}: {response}")
            validated_response, scores = validator.validate_response(response)
            print(f"Validated: {validated_response}")
            print(f"Scores: {scores.to_dict()}")
            print(f"Overall: {scores.get_overall_score():.3f}")

    def _test_sqlite_fallback(self):
        """Test SQLite fallback functionality"""
        print("\n💾 TESTING SQLITE FALLBACK")
        print("=" * 50)
        
        try:
            from chatbot.sqlite_fallback import SQLiteFallback
            
            # Initialize SQLite fallback
            sqlite_fallback = SQLiteFallback()
            
            # Test connection
            print("1. Testing connection...")
            connection_status = sqlite_fallback.test_connection()
            
            if connection_status["connected"]:
                print("✅ SQLite backup database connected successfully")
                print(f"   Documents: {connection_status['document_count']}")
                print(f"   Chunks: {connection_status['chunk_count']}")
                print(f"   Last backup: {connection_status['last_backup']}")
            else:
                print("❌ Failed to connect to SQLite backup")
                print(f"   Error: {connection_status.get('error', 'Unknown error')}")
                return
            
            # Test search functionality
            print("\n2. Testing search functionality...")
            test_queries = [
                "machine learning",
                "neural network", 
                "artificial intelligence"
            ]
            
            for query in test_queries:
                print(f"\n   Searching for: '{query}'")
                
                # Search documents
                documents = sqlite_fallback.search_documents(query, limit=2)
                print(f"   Documents found: {len(documents)}")
                
                # Search chunks
                chunks = sqlite_fallback.search_text_chunks(query, limit=2)
                print(f"   Chunks found: {len(chunks)}")
                
                if documents or chunks:
                    print("   ✅ Found relevant data")
                else:
                    print("   ⚠️  No relevant data found")
            
            # Test fallback response generation
            print("\n3. Testing fallback response generation...")
            test_query = "machine learning patents"
            
            response = sqlite_fallback.generate_fallback_response(test_query)
            print(f"\n   Query: '{test_query}'")
            print(f"   Response: {response[:200]}...")
            
            if response and "backup database" in response:
                print("   ✅ Fallback response generated successfully")
            else:
                print("   ⚠️  Fallback response may be incomplete")
            
            print("\n✅ SQLite Fallback Test Complete")
            
        except Exception as e:
            print(f"❌ SQLite fallback test failed: {e}")

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Patent Analysis Pipeline")
    parser.add_argument("--mode", choices=["interactive", "filter", "upload", "chat", "eval", "pipeline"], 
                       default="interactive", help="Pipeline mode")
    parser.add_argument("--source", default="hupd_processed", help="Source directory for filtering")
    parser.add_argument("--output", default="hupd_processed", help="Output directory for filtering")
    parser.add_argument("--no-guardrails", action="store_true", help="Run chatbot without guardrails validation")
    parser.add_argument("--web-interface", action="store_true", help="Run chatbot with Gradio web interface")
    args = parser.parse_args()
    
    orchestrator = PatentPipelineOrchestrator()
    
    if args.mode == "interactive":
        orchestrator.interactive_menu()
    elif args.mode == "filter":
        orchestrator.run_filtering(args.source, args.output)
    elif args.mode == "upload":
        orchestrator.run_upload(args.output)
    elif args.mode == "chat":
        orchestrator.run_chatbot(
            with_guardrails=not args.no_guardrails,
            web_interface=args.web_interface
        )
    elif args.mode == "eval":
        orchestrator.run_evaluation()
    elif args.mode == "pipeline":
        orchestrator.run_full_pipeline()

if __name__ == "__main__":
    main() 