#!/usr/bin/env python3
"""
Test Neo4j Backup for LightRAG

This script tests whether Neo4j is actually backing up LightRAG data
and provides direct Neo4j query capabilities for verification.
"""

import logging
from neo4j import GraphDatabase
from typing import List, Dict, Any
import json
import time

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class Neo4jBackupTester:
    """Test Neo4j backup and provide direct query capabilities"""
    
    def __init__(self, uri: str = "bolt://localhost:7687", username: str = "neo4j", password: str = "password"):
        self.uri = uri
        self.username = username
        self.password = password
        self.driver = None
        
    def connect(self) -> bool:
        """Connect to Neo4j database"""
        try:
            self.driver = GraphDatabase.driver(self.uri, auth=(self.username, self.password))
            # Test connection
            with self.driver.session() as session:
                result = session.run("RETURN 1 as test")
                test_value = result.single()["test"]
                if test_value == 1:
                    logger.info("✅ Successfully connected to Neo4j")
                    return True
                else:
                    logger.error("❌ Neo4j connection test failed")
                    return False
        except Exception as e:
            logger.error(f"❌ Failed to connect to Neo4j: {e}")
            return False
    
    def disconnect(self):
        """Disconnect from Neo4j"""
        if self.driver:
            self.driver.close()
            logger.info("Disconnected from Neo4j")
    
    def get_database_stats(self) -> Dict[str, Any]:
        """Get database statistics"""
        try:
            with self.driver.session() as session:
                # Get node counts
                node_query = "MATCH (n) RETURN labels(n) as labels, count(n) as count"
                node_result = session.run(node_query)
                node_stats = {}
                for record in node_result:
                    labels = record["labels"]
                    count = record["count"]
                    if labels:
                        node_stats[str(labels)] = count
                
                # Get relationship counts
                rel_query = "MATCH ()-[r]->() RETURN type(r) as type, count(r) as count"
                rel_result = session.run(rel_query)
                rel_stats = {}
                for record in rel_result:
                    rel_type = record["type"]
                    count = record["count"]
                    if rel_type:
                        rel_stats[rel_type] = count
                
                # Get total counts
                total_nodes = session.run("MATCH (n) RETURN count(n) as count").single()["count"]
                total_rels = session.run("MATCH ()-[r]->() RETURN count(r) as count").single()["count"]
                
                return {
                    "total_nodes": total_nodes,
                    "total_relationships": total_rels,
                    "node_types": node_stats,
                    "relationship_types": rel_stats
                }
        except Exception as e:
            logger.error(f"Error getting database stats: {e}")
            return {"error": str(e)}
    
    def find_document_nodes(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Find document nodes in the graph"""
        try:
            with self.driver.session() as session:
                query = """
                MATCH (d:Document)
                RETURN d.id as id, d.title as title, d.content as content
                LIMIT $limit
                """
                result = session.run(query, limit=limit)
                documents = []
                for record in result:
                    documents.append({
                        "id": record["id"],
                        "title": record["title"],
                        "content": record["content"][:200] + "..." if record["content"] and len(record["content"]) > 200 else record["content"]
                    })
                return documents
        except Exception as e:
            logger.error(f"Error finding document nodes: {e}")
            return []
    
    def search_documents_by_content(self, search_term: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Search documents by content"""
        try:
            with self.driver.session() as session:
                query = """
                MATCH (d:Document)
                WHERE d.content CONTAINS $search_term OR d.title CONTAINS $search_term
                RETURN d.id as id, d.title as title, d.content as content
                LIMIT $limit
                """
                result = session.run(query, search_term=search_term, limit=limit)
                documents = []
                for record in result:
                    documents.append({
                        "id": record["id"],
                        "title": record["title"],
                        "content": record["content"][:300] + "..." if record["content"] and len(record["content"]) > 300 else record["content"]
                    })
                return documents
        except Exception as e:
            logger.error(f"Error searching documents: {e}")
            return []
    
    def get_document_relationships(self, doc_id: str) -> Dict[str, Any]:
        """Get relationships for a specific document"""
        try:
            with self.driver.session() as session:
                query = """
                MATCH (d:Document {id: $doc_id})-[r]-(related)
                RETURN type(r) as relationship_type, labels(related) as related_labels, related.id as related_id
                """
                result = session.run(query, doc_id=doc_id)
                relationships = []
                for record in result:
                    relationships.append({
                        "type": record["relationship_type"],
                        "related_labels": record["related_labels"],
                        "related_id": record["related_id"]
                    })
                return relationships
        except Exception as e:
            logger.error(f"Error getting document relationships: {e}")
            return []
    
    def test_lightrag_backup(self) -> Dict[str, Any]:
        """Test if LightRAG data is actually backed up in Neo4j"""
        logger.info("🔍 Testing LightRAG backup in Neo4j...")
        
        stats = self.get_database_stats()
        if "error" in stats:
            return {"success": False, "error": stats["error"]}
        
        # Check if we have document nodes
        documents = self.find_document_nodes(5)
        
        # Check if we have relationships
        has_relationships = stats["total_relationships"] > 0
        
        # Check for specific LightRAG patterns
        with self.driver.session() as session:
            # Check for chunk nodes (LightRAG creates these)
            chunk_query = "MATCH (c:Chunk) RETURN count(c) as count"
            chunk_count = session.run(chunk_query).single()["count"]
            
            # Check for embedding nodes
            embedding_query = "MATCH (e:Embedding) RETURN count(e) as count"
            embedding_count = session.run(embedding_query).single()["count"]
        
        backup_status = {
            "success": True,
            "total_nodes": stats["total_nodes"],
            "total_relationships": stats["total_relationships"],
            "document_count": len(documents),
            "chunk_count": chunk_count,
            "embedding_count": embedding_count,
            "has_relationships": has_relationships,
            "node_types": stats["node_types"],
            "relationship_types": stats["relationship_types"],
            "sample_documents": documents
        }
        
        # Determine if backup is working
        if stats["total_nodes"] > 0 and has_relationships:
            backup_status["backup_working"] = True
            logger.info("✅ Neo4j appears to be backing up LightRAG data")
        else:
            backup_status["backup_working"] = False
            logger.warning("⚠️  Neo4j may not be properly backing up LightRAG data")
        
        return backup_status
    
    def interactive_test(self):
        """Interactive test mode"""
        print("\n🔬 NEO4J BACKUP TESTER")
        print("=" * 50)
        
        if not self.connect():
            print("❌ Cannot connect to Neo4j. Please ensure Neo4j is running.")
            return
        
        try:
            while True:
                print("\nOptions:")
                print("1. Test LightRAG backup")
                print("2. Show database stats")
                print("3. Search documents")
                print("4. Find document relationships")
                print("5. Exit")
                
                choice = input("\nSelect option (1-5): ").strip()
                
                if choice == "1":
                    result = self.test_lightrag_backup()
                    print(f"\n📊 Backup Test Results:")
                    print(f"   Total Nodes: {result['total_nodes']}")
                    print(f"   Total Relationships: {result['total_relationships']}")
                    print(f"   Document Count: {result['document_count']}")
                    print(f"   Chunk Count: {result['chunk_count']}")
                    print(f"   Embedding Count: {result['embedding_count']}")
                    print(f"   Backup Working: {'✅ YES' if result['backup_working'] else '❌ NO'}")
                    
                    if result['sample_documents']:
                        print(f"\n📄 Sample Documents:")
                        for i, doc in enumerate(result['sample_documents'][:3], 1):
                            print(f"   {i}. {doc['title']} (ID: {doc['id']})")
                
                elif choice == "2":
                    stats = self.get_database_stats()
                    print(f"\n📈 Database Statistics:")
                    print(f"   Total Nodes: {stats['total_nodes']}")
                    print(f"   Total Relationships: {stats['total_relationships']}")
                    print(f"   Node Types: {stats['node_types']}")
                    print(f"   Relationship Types: {stats['relationship_types']}")
                
                elif choice == "3":
                    search_term = input("Enter search term: ").strip()
                    if search_term:
                        results = self.search_documents_by_content(search_term)
                        print(f"\n🔍 Search Results for '{search_term}':")
                        for i, doc in enumerate(results, 1):
                            print(f"   {i}. {doc['title']}")
                            print(f"      Content: {doc['content'][:100]}...")
                            print()
                
                elif choice == "4":
                    doc_id = input("Enter document ID: ").strip()
                    if doc_id:
                        relationships = self.get_document_relationships(doc_id)
                        print(f"\n🔗 Relationships for document {doc_id}:")
                        for rel in relationships:
                            print(f"   - {rel['type']} -> {rel['related_labels']} (ID: {rel['related_id']})")
                
                elif choice == "5":
                    break
                
                else:
                    print("❌ Invalid option")
        
        finally:
            self.disconnect()

def main():
    """Main function"""
    tester = Neo4jBackupTester()
    tester.interactive_test()

if __name__ == "__main__":
    main() 