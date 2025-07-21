#!/usr/bin/env python3
"""
Neo4j Fallback for Patent Chatbot

This module provides direct Neo4j query capabilities as a fallback
when LightRAG server is unavailable.
"""

import logging
from neo4j import GraphDatabase
from typing import List, Dict, Any, Optional
import json
import time
from datetime import datetime

logger = logging.getLogger(__name__)

class Neo4jFallback:
    """Direct Neo4j query fallback for chatbot"""
    
    def __init__(self, uri: str = "bolt://localhost:7687", username: str = "neo4j", password: str = "password"):
        self.uri = uri
        self.username = username
        self.password = password
        self.driver = None
        self.connected = False
        
    def connect(self) -> bool:
        """Connect to Neo4j database"""
        try:
            self.driver = GraphDatabase.driver(self.uri, auth=(self.username, self.password))
            # Test connection
            with self.driver.session() as session:
                result = session.run("RETURN 1 as test")
                test_value = result.single()["test"]
                if test_value == 1:
                    self.connected = True
                    logger.info("✅ Successfully connected to Neo4j fallback")
                    return True
                else:
                    logger.error("❌ Neo4j fallback connection test failed")
                    return False
        except Exception as e:
            logger.error(f"❌ Failed to connect to Neo4j fallback: {e}")
            return False
    
    def disconnect(self):
        """Disconnect from Neo4j"""
        if self.driver:
            self.driver.close()
            self.connected = False
            logger.info("Disconnected from Neo4j fallback")
    
    def is_available(self) -> bool:
        """Check if Neo4j fallback is available"""
        if not self.connected:
            return self.connect()
        return True
    
    def search_documents(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Search documents using Neo4j queries"""
        if not self.is_available():
            return []
        
        try:
            with self.driver.session() as session:
                # Search in document nodes (actual node type in database)
                search_query = """
                MATCH (d:document)
                WHERE d.description CONTAINS $search_term OR d.file_path CONTAINS $search_term
                RETURN d.entity_id as id, d.file_path as title, d.description as content
                LIMIT $limit
                """
                result = session.run(search_query, search_term=query, limit=limit)
                documents = []
                for record in result:
                    documents.append({
                        "id": record["id"],
                        "title": record["title"],
                        "content": record["content"]
                    })
                return documents
        except Exception as e:
            logger.error(f"Error searching documents in Neo4j: {e}")
            return []
    
    def get_document_by_id(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """Get a specific document by ID"""
        if not self.is_available():
            return None
        
        try:
            with self.driver.session() as session:
                query = """
                MATCH (d:document {entity_id: $doc_id})
                RETURN d.entity_id as id, d.file_path as title, d.description as content
                """
                result = session.run(query, doc_id=doc_id)
                record = result.single()
                if record:
                    return {
                        "id": record["id"],
                        "title": record["title"],
                        "content": record["content"]
                    }
                return None
        except Exception as e:
            logger.error(f"Error getting document by ID: {e}")
            return None
    
    def get_related_documents(self, doc_id: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Get documents related to a specific document"""
        if not self.is_available():
            return []
        
        try:
            with self.driver.session() as session:
                query = """
                MATCH (d:document {entity_id: $doc_id})-[r]-(related:document)
                RETURN related.entity_id as id, related.file_path as title, related.description as content, type(r) as relationship
                LIMIT $limit
                """
                result = session.run(query, doc_id=doc_id, limit=limit)
                documents = []
                for record in result:
                    documents.append({
                        "id": record["id"],
                        "title": record["title"],
                        "content": record["content"],
                        "relationship": record["relationship"]
                    })
                return documents
        except Exception as e:
            logger.error(f"Error getting related documents: {e}")
            return []
    
    def get_document_statistics(self) -> Dict[str, Any]:
        """Get document statistics from Neo4j"""
        if not self.is_available():
            return {"error": "Not connected to Neo4j"}
        
        try:
            with self.driver.session() as session:
                # Get total document count (using actual node type)
                doc_count = session.run("MATCH (d:document) RETURN count(d) as count").single()["count"]
                
                # Get all node types to see what's actually in the database
                node_types = session.run("""
                    MATCH (n)
                    RETURN labels(n) as labels, count(n) as count
                """)
                node_stats = {}
                for record in node_types:
                    labels = record["labels"]
                    count = record["count"]
                    if labels:
                        node_stats[str(labels)] = count
                
                # Get document types/categories if they exist
                try:
                    categories = session.run("""
                        MATCH (d:document)
                        WHERE d.entity_type IS NOT NULL
                        RETURN d.entity_type as category, count(d) as count
                    """)
                    category_stats = {}
                    for record in categories:
                        category = record["category"]
                        count = record["count"]
                        if category:
                            category_stats[category] = count
                except Exception:
                    category_stats = {}
                
                return {
                    "total_documents": doc_count,
                    "categories": category_stats,
                    "all_node_types": node_stats,
                    "source": "Neo4j Fallback"
                }
        except Exception as e:
            logger.error(f"Error getting document statistics: {e}")
            return {"error": str(e)}
    
    def generate_fallback_response(self, query: str) -> str:
        """Generate a fallback response using Neo4j data"""
        if not self.is_available():
            return "I apologize, but I cannot access the patent database at the moment. Please try again later."
        
        try:
            # Search for relevant documents
            documents = self.search_documents(query, limit=5)
            
            if not documents:
                return "I couldn't find any patents matching your query in the database."
            
            # Generate a simple response based on found documents
            response_parts = []
            response_parts.append(f"I found {len(documents)} relevant patents:")
            
            for i, doc in enumerate(documents[:3], 1):
                title = doc.get('title', 'Unknown Title')
                content_preview = doc.get('content', '')[:150] + "..." if doc.get('content') else "No content available"
                response_parts.append(f"\n{i}. {title}")
                response_parts.append(f"   {content_preview}")
            
            if len(documents) > 3:
                response_parts.append(f"\n... and {len(documents) - 3} more patents.")
            
            response_parts.append("\n\nNote: This response was generated using the Neo4j fallback system.")
            
            return "\n".join(response_parts)
            
        except Exception as e:
            logger.error(f"Error generating fallback response: {e}")
            return "I encountered an error while searching the patent database. Please try again later."
    
    def test_connection(self) -> Dict[str, Any]:
        """Test Neo4j connection and basic functionality"""
        test_results = {
            "connected": False,
            "document_count": 0,
            "search_working": False,
            "node_types": {},
            "error": None
        }
        
        try:
            if not self.connect():
                test_results["error"] = "Failed to connect to Neo4j"
                return test_results
            
            test_results["connected"] = True
            
            # Test document count and get node types
            stats = self.get_document_statistics()
            if "error" not in stats:
                test_results["document_count"] = stats.get("total_documents", 0)
                test_results["node_types"] = stats.get("all_node_types", {})
            
            # Test search functionality only if documents exist
            if test_results["document_count"] > 0:
                test_search = self.search_documents("patent", limit=1)
                test_results["search_working"] = len(test_search) > 0
            else:
                test_results["search_working"] = False
            
            return test_results
            
        except Exception as e:
            test_results["error"] = str(e)
            return test_results
        finally:
            self.disconnect()

def test_neo4j_fallback():
    """Test the Neo4j fallback functionality"""
    print("🔬 Testing Neo4j Fallback...")
    
    fallback = Neo4jFallback()
    test_results = fallback.test_connection()
    
    print(f"\n📊 Test Results:")
    print(f"   Connected: {'✅ YES' if test_results['connected'] else '❌ NO'}")
    print(f"   Document Count: {test_results['document_count']}")
    print(f"   Search Working: {'✅ YES' if test_results['search_working'] else '❌ NO'}")
    
    if test_results['error']:
        print(f"   Error: {test_results['error']}")
    
    # Test fallback response generation
    if test_results['connected']:
        print(f"\n🧪 Testing fallback response generation...")
        test_query = "machine learning"
        response = fallback.generate_fallback_response(test_query)
        print(f"Query: '{test_query}'")
        print(f"Response: {response[:200]}...")

if __name__ == "__main__":
    test_neo4j_fallback() 