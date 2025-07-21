#!/usr/bin/env python3
"""
Neo4j Backup Monitor

This script monitors LightRAG JSON files and syncs them to Neo4j,
extracting entities and relationships from patent documents.
"""

import json
import logging
import time
import hashlib
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Set
import re

# Neo4j imports
try:
    from neo4j import GraphDatabase
except ImportError:
    print("❌ Neo4j driver not installed. Install with: pip install neo4j")
    exit(1)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class Neo4jBackupMonitor:
    """Monitor LightRAG files and sync to Neo4j with entity extraction"""
    
    def __init__(self, 
                 lightrag_storage_dir: str = "rag_storage",
                 neo4j_uri: str = "bolt://localhost:7687",
                 neo4j_user: str = "neo4j",
                 neo4j_password: str = "password",
                 check_interval: int = 60):
        self.lightrag_storage_dir = Path(lightrag_storage_dir)
        self.neo4j_uri = neo4j_uri
        self.neo4j_user = neo4j_user
        self.neo4j_password = neo4j_password
        self.check_interval = check_interval
        self.running = False
        
        # Track processed file hashes
        self.processed_hashes = {}
        
        # Entity extraction patterns
        self.entity_patterns = {
            'patent_number': r'US\d+[A-Z]?\d*',
            'application_number': r'\d{8,}',
            'organization': r'\b[A-Z][a-zA-Z\s&]+(?:Corp|Inc|LLC|Ltd|Company|Corporation|Limited)\b',
            'technology': r'\b(?:AI|ML|machine learning|artificial intelligence|neural network|blockchain|IoT|internet of things)\b',
            'date': r'\b(?:19|20)\d{2}-\d{2}-\d{2}\b'
        }
        
        # Initialize Neo4j connection
        self.driver = None
        self.init_neo4j_connection()
        
    def init_neo4j_connection(self):
        """Initialize Neo4j connection"""
        try:
            self.driver = GraphDatabase.driver(
                self.neo4j_uri, 
                auth=(self.neo4j_user, self.neo4j_password)
            )
            # Test connection
            with self.driver.session() as session:
                session.run("RETURN 1")
            logger.info("✅ Neo4j connection established")
        except Exception as e:
            logger.error(f"❌ Failed to connect to Neo4j: {e}")
            self.driver = None
    
    def get_file_hash(self, file_path: Path) -> str:
        """Get MD5 hash of a file"""
        try:
            with open(file_path, 'rb') as f:
                return hashlib.md5(f.read()).hexdigest()
        except Exception:
            return ""
    
    def has_file_changed(self, file_path: Path) -> bool:
        """Check if a file has changed since last processing"""
        current_hash = self.get_file_hash(file_path)
        last_hash = self.processed_hashes.get(str(file_path), "")
        
        if current_hash != last_hash:
            self.processed_hashes[str(file_path)] = current_hash
            return True
        return False
    
    def extract_entities(self, text: str) -> Dict[str, List[str]]:
        """Extract entities from patent text"""
        entities = {}
        
        for entity_type, pattern in self.entity_patterns.items():
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                entities[entity_type] = list(set(matches))  # Remove duplicates
        
        return entities
    
    def create_document_node(self, doc_id: str, content: str, file_path: str = "unknown_source"):
        """Create a document node in Neo4j"""
        try:
            with self.driver.session() as session:
                # Extract entities from content
                entities = self.extract_entities(content)
                
                # Create document node
                query = """
                MERGE (d:document {entity_id: $doc_id})
                SET d.description = $content,
                    d.file_path = $file_path,
                    d.entity_type = 'patent_document',
                    d.last_updated = datetime()
                """
                session.run(query, doc_id=doc_id, content=content, file_path=file_path)
                
                # Create entity nodes and relationships
                for entity_type, entity_list in entities.items():
                    for entity in entity_list:
                        # Create entity node
                        entity_query = f"""
                        MERGE (e:{entity_type} {{name: $entity_name}})
                        SET e.entity_type = $entity_type,
                            e.last_updated = datetime()
                        """
                        session.run(entity_query, entity_name=entity, entity_type=entity_type)
                        
                        # Create relationship
                        rel_query = f"""
                        MATCH (d:document {{entity_id: $doc_id}})
                        MATCH (e:{entity_type} {{name: $entity_name}})
                        MERGE (d)-[r:CONTAINS]->(e)
                        SET r.last_updated = datetime()
                        """
                        session.run(rel_query, doc_id=doc_id, entity_name=entity)
                
                logger.info(f"✅ Created document node: {doc_id} with {len(entities)} entity types")
                return True
                
        except Exception as e:
            logger.error(f"❌ Failed to create document node {doc_id}: {e}")
            return False
    
    def sync_documents_to_neo4j(self, documents_data: Dict[str, Any]) -> int:
        """Sync documents from LightRAG to Neo4j"""
        if not self.driver:
            logger.error("❌ Neo4j connection not available")
            return 0
        
        try:
            synced_count = 0
            
            for doc_id, doc_data in documents_data.items():
                content = doc_data.get('content', '')
                if content:
                    if self.create_document_node(doc_id, content):
                        synced_count += 1
            
            logger.info(f"✅ Synced {synced_count} documents to Neo4j")
            return synced_count
            
        except Exception as e:
            logger.error(f"❌ Failed to sync documents to Neo4j: {e}")
            return 0
    
    def get_neo4j_stats(self) -> Dict[str, Any]:
        """Get Neo4j statistics"""
        if not self.driver:
            return {"error": "Neo4j not connected"}
        
        try:
            with self.driver.session() as session:
                # Get document count
                doc_count = session.run("MATCH (d:document) RETURN count(d) as count").single()["count"]
                
                # Get entity counts
                entity_counts = {}
                for entity_type in self.entity_patterns.keys():
                    try:
                        count = session.run(f"MATCH (n:{entity_type}) RETURN count(n) as count").single()["count"]
                        entity_counts[entity_type] = count
                    except:
                        entity_counts[entity_type] = 0
                
                # Get relationship count
                rel_count = session.run("MATCH ()-[r:CONTAINS]->() RETURN count(r) as count").single()["count"]
                
                # Get last update time
                last_update = session.run("""
                    MATCH (d:document) 
                    RETURN max(d.last_updated) as last_update
                """).single()["last_update"]
                
                return {
                    "document_count": doc_count,
                    "entity_counts": entity_counts,
                    "relationship_count": rel_count,
                    "last_update": str(last_update) if last_update else "Never"
                }
                
        except Exception as e:
            logger.error(f"❌ Failed to get Neo4j stats: {e}")
            return {"error": str(e)}
    
    def monitor_and_sync(self):
        """Monitor LightRAG files and sync to Neo4j"""
        logger.info("🔍 Starting Neo4j backup monitor...")
        
        while self.running:
            try:
                # Check LightRAG documents file
                docs_file = self.lightrag_storage_dir / "kv_store_full_docs.json"
                
                if docs_file.exists() and self.has_file_changed(docs_file):
                    logger.info("📄 Detected changes in LightRAG documents, syncing to Neo4j...")
                    
                    # Read LightRAG documents
                    with open(docs_file, 'r') as f:
                        documents_data = json.load(f)
                    
                    # Sync to Neo4j
                    synced_count = self.sync_documents_to_neo4j(documents_data)
                    
                    if synced_count > 0:
                        logger.info(f"📦 Synced {synced_count} documents to Neo4j")
                    
                # Wait before next check
                time.sleep(self.check_interval)
                
            except Exception as e:
                logger.error(f"❌ Error in Neo4j backup monitor: {e}")
                time.sleep(self.check_interval)
    
    def start_monitoring(self):
        """Start the Neo4j backup monitoring"""
        if not self.driver:
            logger.error("❌ Cannot start monitoring - Neo4j not connected")
            return
        
        self.running = True
        logger.info("🚀 Neo4j backup monitor started")
        self.monitor_and_sync()
    
    def stop_monitoring(self):
        """Stop the Neo4j backup monitoring"""
        self.running = False
        if self.driver:
            self.driver.close()
        logger.info("🛑 Neo4j backup monitor stopped")

def main():
    """Main function to run the Neo4j backup monitor"""
    print("🔗 Neo4j Backup Monitor")
    print("=" * 50)
    
    monitor = Neo4jBackupMonitor()
    
    if not monitor.driver:
        print("❌ Failed to connect to Neo4j. Please ensure Neo4j is running.")
        return
    
    try:
        # Show initial stats
        stats = monitor.get_neo4j_stats()
        print(f"\n📊 Neo4j Statistics:")
        print(f"   Documents: {stats.get('document_count', 0)}")
        print(f"   Relationships: {stats.get('relationship_count', 0)}")
        print(f"   Last Update: {stats.get('last_update', 'Never')}")
        
        if 'entity_counts' in stats:
            print(f"   Entity Types:")
            for entity_type, count in stats['entity_counts'].items():
                if count > 0:
                    print(f"     {entity_type}: {count}")
        
        print(f"\n🔄 Monitoring LightRAG files every 60 seconds...")
        print("Press Ctrl+C to stop")
        
        # Start monitoring
        monitor.start_monitoring()
        
    except KeyboardInterrupt:
        print("\n🛑 Stopping Neo4j backup monitor...")
        monitor.stop_monitoring()
        
        # Show final stats
        stats = monitor.get_neo4j_stats()
        print(f"\n📊 Final Neo4j Statistics:")
        print(f"   Documents: {stats.get('document_count', 0)}")
        print(f"   Relationships: {stats.get('relationship_count', 0)}")
        print(f"   Last Update: {stats.get('last_update', 'Never')}")

if __name__ == "__main__":
    main() 