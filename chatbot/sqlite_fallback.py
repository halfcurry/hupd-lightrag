#!/usr/bin/env python3
"""
SQLite Fallback for Patent Chatbot

This module provides direct SQLite query capabilities as a fallback
when LightRAG server is busy or unavailable.
"""

import logging
import sqlite3
import json
from typing import List, Dict, Any, Optional
from datetime import datetime

logger = logging.getLogger(__name__)

class SQLiteFallback:
    """Direct SQLite query fallback for chatbot"""
    
    def __init__(self, backup_db_path: str = "lightrag_backup.db"):
        self.backup_db_path = backup_db_path
        self.connected = False
        
    def connect(self) -> bool:
        """Connect to SQLite backup database"""
        try:
            # Test connection by checking if database exists and has tables
            conn = sqlite3.connect(self.backup_db_path)
            cursor = conn.cursor()
            
            # Check if tables exist
            cursor.execute("""
                SELECT name FROM sqlite_master 
                WHERE type='table' AND name IN ('full_docs', 'text_chunks', 'doc_status')
            """)
            tables = cursor.fetchall()
            
            if len(tables) >= 2:  # At least full_docs and text_chunks should exist
                self.connected = True
                conn.close()
                logger.info("✅ Successfully connected to SQLite backup")
                return True
            else:
                logger.error("❌ SQLite backup database missing required tables")
                return False
                
        except Exception as e:
            logger.error(f"❌ Failed to connect to SQLite backup: {e}")
            return False
    
    def is_available(self) -> bool:
        """Check if SQLite fallback is available"""
        if not self.connected:
            return self.connect()
        return True
    
    def get_backup_stats(self) -> Dict[str, Any]:
        """Get statistics about the backup database"""
        try:
            conn = sqlite3.connect(self.backup_db_path)
            cursor = conn.cursor()
            
            # Get record counts
            cursor.execute("SELECT COUNT(*) FROM full_docs")
            full_docs_count = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM text_chunks")
            text_chunks_count = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM doc_status")
            doc_status_count = cursor.fetchone()[0]
            
            # Get last backup time
            cursor.execute("SELECT MAX(last_updated) FROM full_docs")
            last_backup = cursor.fetchone()[0]
            
            conn.close()
            
            return {
                "full_docs_count": full_docs_count,
                "text_chunks_count": text_chunks_count,
                "doc_status_count": doc_status_count,
                "last_backup": last_backup,
                "backup_db_path": self.backup_db_path
            }
            
        except Exception as e:
            logger.error(f"❌ Failed to get backup stats: {e}")
            return {"error": str(e)}
    
    def search_documents(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Search documents in SQLite backup"""
        try:
            conn = sqlite3.connect(self.backup_db_path)
            cursor = conn.cursor()
            
            # Search in full documents
            cursor.execute("""
                SELECT doc_id, content, last_updated 
                FROM full_docs 
                WHERE content LIKE ? 
                ORDER BY last_updated DESC 
                LIMIT ?
            """, (f"%{query}%", limit))
            
            results = []
            for row in cursor.fetchall():
                results.append({
                    "doc_id": row[0],
                    "content": row[1],
                    "last_updated": row[2],
                    "source": "sqlite_backup"
                })
            
            conn.close()
            logger.info(f"✅ Found {len(results)} documents in SQLite backup")
            return results
            
        except Exception as e:
            logger.error(f"❌ Failed to search SQLite backup: {e}")
            return []
    
    def search_text_chunks(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Search text chunks in SQLite backup"""
        try:
            conn = sqlite3.connect(self.backup_db_path)
            cursor = conn.cursor()
            
            # Search in text chunks
            cursor.execute("""
                SELECT chunk_id, content, doc_id, last_updated 
                FROM text_chunks 
                WHERE content LIKE ? 
                ORDER BY last_updated DESC 
                LIMIT ?
            """, (f"%{query}%", limit))
            
            results = []
            for row in cursor.fetchall():
                results.append({
                    "chunk_id": row[0],
                    "content": row[1],
                    "doc_id": row[2],
                    "last_updated": row[3],
                    "source": "sqlite_backup"
                })
            
            conn.close()
            logger.info(f"✅ Found {len(results)} text chunks in SQLite backup")
            return results
            
        except Exception as e:
            logger.error(f"❌ Failed to search SQLite text chunks: {e}")
            return []
    
    def generate_fallback_response(self, query: str) -> str:
        """Generate a response using SQLite backup data"""
        try:
            # Search for relevant documents
            documents = self.search_documents(query, limit=5)
            chunks = self.search_text_chunks(query, limit=5)
            
            if not documents and not chunks:
                return f"I apologize, but I couldn't find any relevant patent information in my backup database for '{query}'. The LightRAG server may be busy, and my backup data doesn't contain information about this topic."
            
            # Combine results
            all_results = documents + chunks
            
            # Create a response based on found data
            response = f"Based on my backup database, I found {len(all_results)} relevant patent documents:\n\n"
            
            for i, result in enumerate(all_results[:3], 1):
                content = result.get('content', '')
                # Truncate content for readability
                if len(content) > 200:
                    content = content[:200] + "..."
                
                response += f"{i}. {content}\n\n"
            
            if len(all_results) > 3:
                response += f"... and {len(all_results) - 3} more documents.\n\n"
            
            response += "Note: This response was generated from my backup database as the LightRAG server is currently busy."
            
            return response
            
        except Exception as e:
            logger.error(f"❌ Error generating SQLite fallback response: {e}")
            return f"I apologize, but I encountered an error while searching my backup database for '{query}'. The LightRAG server may be busy, and I'm unable to access my backup data at the moment."
    
    def test_connection(self) -> Dict[str, Any]:
        """Test SQLite connection and return status"""
        try:
            stats = self.get_backup_stats()
            
            if "error" in stats:
                return {
                    "connected": False,
                    "error": stats["error"],
                    "document_count": 0,
                    "backup_working": False
                }
            
            return {
                "connected": True,
                "document_count": stats.get("full_docs_count", 0),
                "chunk_count": stats.get("text_chunks_count", 0),
                "last_backup": stats.get("last_backup", "Unknown"),
                "backup_working": True
            }
            
        except Exception as e:
            return {
                "connected": False,
                "error": str(e),
                "document_count": 0,
                "backup_working": False
            } 