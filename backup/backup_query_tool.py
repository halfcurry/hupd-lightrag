#!/usr/bin/env python3
"""
LightRAG Backup Query Tool

Simple tool to query backed up LightRAG data from the SQLite database.
"""

import sqlite3
import json
from pathlib import Path
from typing import List, Dict, Any

class BackupQueryTool:
    """Query tool for LightRAG backup database"""
    
    def __init__(self, backup_db_path: str = "lightrag_backup.db"):
        self.backup_db_path = Path(backup_db_path)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get backup database statistics"""
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
            cursor.execute("SELECT MAX(timestamp) FROM backup_log")
            last_backup = cursor.fetchone()[0]
            
            conn.close()
            
            return {
                "full_docs_count": full_docs_count,
                "text_chunks_count": text_chunks_count,
                "doc_status_count": doc_status_count,
                "last_backup": last_backup,
                "backup_db_path": str(self.backup_db_path)
            }
            
        except Exception as e:
            return {"error": str(e)}
    
    def search_documents(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Search documents by content"""
        try:
            conn = sqlite3.connect(self.backup_db_path)
            cursor = conn.cursor()
            
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
                    "content": row[1][:200] + "..." if len(row[1]) > 200 else row[1],
                    "last_updated": row[2]
                })
            
            conn.close()
            return results
            
        except Exception as e:
            print(f"❌ Error searching documents: {e}")
            return []
    
    def get_document_by_id(self, doc_id: str) -> Dict[str, Any]:
        """Get a specific document by ID"""
        try:
            conn = sqlite3.connect(self.backup_db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT doc_id, content, last_updated 
                FROM full_docs 
                WHERE doc_id = ?
            """, (doc_id,))
            
            row = cursor.fetchone()
            conn.close()
            
            if row:
                return {
                    "doc_id": row[0],
                    "content": row[1],
                    "last_updated": row[2]
                }
            else:
                return {"error": "Document not found"}
                
        except Exception as e:
            return {"error": str(e)}
    
    def get_recent_documents(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get most recently updated documents"""
        try:
            conn = sqlite3.connect(self.backup_db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT doc_id, content, last_updated 
                FROM full_docs 
                ORDER BY last_updated DESC 
                LIMIT ?
            """, (limit,))
            
            results = []
            for row in cursor.fetchall():
                results.append({
                    "doc_id": row[0],
                    "content": row[1][:200] + "..." if len(row[1]) > 200 else row[1],
                    "last_updated": row[2]
                })
            
            conn.close()
            return results
            
        except Exception as e:
            print(f"❌ Error getting recent documents: {e}")
            return []

def main():
    """Interactive query tool"""
    print("🔍 LightRAG Backup Query Tool")
    print("=" * 40)
    
    tool = BackupQueryTool()
    
    # Show stats
    stats = tool.get_stats()
    if "error" not in stats:
        print(f"\n📊 Backup Database Statistics:")
        print(f"   Full Documents: {stats['full_docs_count']}")
        print(f"   Text Chunks: {stats['text_chunks_count']}")
        print(f"   Document Status: {stats['doc_status_count']}")
        print(f"   Last Backup: {stats['last_backup']}")
    else:
        print(f"❌ Error: {stats['error']}")
        return
    
    while True:
        print(f"\nOptions:")
        print("1. Search documents")
        print("2. Get recent documents")
        print("3. Get document by ID")
        print("4. Show stats")
        print("5. Exit")
        
        choice = input("\nSelect option (1-5): ").strip()
        
        if choice == "1":
            query = input("Enter search term: ").strip()
            if query:
                results = tool.search_documents(query, 5)
                print(f"\n🔍 Found {len(results)} documents:")
                for i, doc in enumerate(results, 1):
                    print(f"\n{i}. {doc['doc_id']}")
                    print(f"   Content: {doc['content']}")
                    print(f"   Updated: {doc['last_updated']}")
        
        elif choice == "2":
            results = tool.get_recent_documents(5)
            print(f"\n📄 Recent Documents:")
            for i, doc in enumerate(results, 1):
                print(f"\n{i}. {doc['doc_id']}")
                print(f"   Content: {doc['content']}")
                print(f"   Updated: {doc['last_updated']}")
        
        elif choice == "3":
            doc_id = input("Enter document ID: ").strip()
            if doc_id:
                doc = tool.get_document_by_id(doc_id)
                if "error" not in doc:
                    print(f"\n📄 Document: {doc['doc_id']}")
                    print(f"   Content: {doc['content']}")
                    print(f"   Updated: {doc['last_updated']}")
                else:
                    print(f"❌ {doc['error']}")
        
        elif choice == "4":
            stats = tool.get_stats()
            print(f"\n📊 Current Statistics:")
            print(f"   Full Documents: {stats['full_docs_count']}")
            print(f"   Text Chunks: {stats['text_chunks_count']}")
            print(f"   Document Status: {stats['doc_status_count']}")
            print(f"   Last Backup: {stats['last_backup']}")
        
        elif choice == "5":
            print("👋 Goodbye!")
            break
        
        else:
            print("❌ Invalid option")

if __name__ == "__main__":
    main() 