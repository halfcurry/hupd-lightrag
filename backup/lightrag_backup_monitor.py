#!/usr/bin/env python3
"""
LightRAG Backup Monitor

This script monitors LightRAG JSON storage files and backs them up to a separate database
without affecting LightRAG's current operations.
"""

import json
import logging
import time
import sqlite3
import threading
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List
import hashlib
import shutil

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class LightRAGBackupMonitor:
    """Monitor and backup LightRAG JSON files to a separate database
    
    Note: LightRAG uses 'lightrag_storage/' directory
    """
    
    def __init__(self, 
                 lightrag_storage_dir: str = "rag_storage",
                 backup_db_path: str = "lightrag_backup.db",
                 check_interval: int = 30):
        self.lightrag_storage_dir = Path(lightrag_storage_dir)
        self.backup_db_path = Path(backup_db_path)
        self.check_interval = check_interval
        self.running = False
        self.monitor_thread = None
        
        # Initialize backup database
        self.init_backup_db()
        
        # Track file hashes to detect changes
        self.file_hashes = {}
        
        # Initialize file hashes from database
        self.init_file_hashes()
    
    def init_backup_db(self):
        """Initialize the backup SQLite database"""
        try:
            conn = sqlite3.connect(self.backup_db_path)
            cursor = conn.cursor()
            
            # Create tables for different types of data
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS full_docs (
                    doc_id TEXT PRIMARY KEY,
                    content TEXT,
                    last_updated TIMESTAMP,
                    file_hash TEXT
                )
            ''')
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS text_chunks (
                    chunk_id TEXT PRIMARY KEY,
                    content TEXT,
                    doc_id TEXT,
                    last_updated TIMESTAMP,
                    file_hash TEXT
                )
            ''')
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS doc_status (
                    doc_id TEXT PRIMARY KEY,
                    status TEXT,
                    last_updated TIMESTAMP,
                    file_hash TEXT
                )
            ''')
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS backup_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TIMESTAMP,
                    action TEXT,
                    file_name TEXT,
                    records_count INTEGER,
                    status TEXT
                )
            ''')
            
            conn.commit()
            conn.close()
            logger.info(f"✅ Backup database initialized: {self.backup_db_path}")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize backup database: {e}")
    
    def init_file_hashes(self):
        """Initialize file hashes from existing database records"""
        try:
            conn = sqlite3.connect(self.backup_db_path)
            cursor = conn.cursor()
            
            # Get the most recent file hash for each table
            tables = ["full_docs", "text_chunks", "doc_status"]
            file_mapping = {
                "full_docs": "kv_store_full_docs.json",
                "text_chunks": "kv_store_text_chunks.json", 
                "doc_status": "kv_store_doc_status.json"
            }
            
            for table in tables:
                cursor.execute(f"""
                    SELECT file_hash FROM {table} 
                    WHERE file_hash IS NOT NULL 
                    ORDER BY last_updated DESC 
                    LIMIT 1
                """)
                result = cursor.fetchone()
                if result and result[0]:
                    file_name = file_mapping[table]
                    file_path = self.lightrag_storage_dir / file_name
                    self.file_hashes[str(file_path)] = result[0]
                    logger.info(f"📄 Loaded hash for {file_name}: {result[0][:16]}...")
            
            conn.close()
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize file hashes: {e}")
    
    def get_file_hash(self, file_path: Path) -> str:
        """Get MD5 hash of a file"""
        try:
            with open(file_path, 'rb') as f:
                return hashlib.md5(f.read()).hexdigest()
        except Exception:
            return ""
    
    def has_file_changed(self, file_path: Path) -> bool:
        """Check if a file has changed since last backup"""
        current_hash = self.get_file_hash(file_path)
        last_hash = self.file_hashes.get(str(file_path), "")
        
        if current_hash != last_hash:
            self.file_hashes[str(file_path)] = current_hash
            return True
        return False
    
    def backup_json_file(self, file_path: Path, table_name: str) -> Dict[str, Any]:
        """Backup a JSON file to the database"""
        try:
            if not file_path.exists():
                return {"success": False, "error": "File not found", "records": 0}
            
            # Check if file has changed
            if not self.has_file_changed(file_path):
                return {"success": True, "message": "No changes detected", "records": 0}
            
            # Read JSON file
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            conn = sqlite3.connect(self.backup_db_path)
            cursor = conn.cursor()
            
            records_count = 0
            current_time = datetime.now().isoformat()
            file_hash = self.get_file_hash(file_path)
            
            if table_name == "full_docs":
                # Backup full documents
                for doc_id, doc_data in data.items():
                    content = doc_data.get('content', '')
                    cursor.execute('''
                        INSERT OR REPLACE INTO full_docs (doc_id, content, last_updated, file_hash)
                        VALUES (?, ?, ?, ?)
                    ''', (doc_id, content, current_time, file_hash))
                    records_count += 1
                    
            elif table_name == "text_chunks":
                # Backup text chunks
                for chunk_id, chunk_data in data.items():
                    content = chunk_data.get('content', '')
                    doc_id = chunk_data.get('doc_id', '')
                    cursor.execute('''
                        INSERT OR REPLACE INTO text_chunks (chunk_id, content, doc_id, last_updated, file_hash)
                        VALUES (?, ?, ?, ?, ?)
                    ''', (chunk_id, content, doc_id, current_time, file_hash))
                    records_count += 1
                    
            elif table_name == "doc_status":
                # Backup document status
                for doc_id, status_data in data.items():
                    status = json.dumps(status_data)
                    cursor.execute('''
                        INSERT OR REPLACE INTO doc_status (doc_id, status, last_updated, file_hash)
                        VALUES (?, ?, ?, ?)
                    ''', (doc_id, status, current_time, file_hash))
                    records_count += 1
            
            # Log the backup
            cursor.execute('''
                INSERT INTO backup_log (timestamp, action, file_name, records_count, status)
                VALUES (?, ?, ?, ?, ?)
            ''', (current_time, f"backup_{table_name}", file_path.name, records_count, "success"))
            
            conn.commit()
            conn.close()
            
            logger.info(f"✅ Backed up {records_count} records from {file_path.name}")
            return {"success": True, "records": records_count}
            
        except Exception as e:
            logger.error(f"❌ Failed to backup {file_path}: {e}")
            return {"success": False, "error": str(e), "records": 0}
    
    def monitor_and_backup(self):
        """Monitor LightRAG files and backup when changes are detected"""
        logger.info("🔍 Starting LightRAG backup monitor...")
        
        while self.running:
            try:
                # Check each JSON file
                json_files = [
                    ("kv_store_full_docs.json", "full_docs"),
                    ("kv_store_text_chunks.json", "text_chunks"),
                    ("kv_store_doc_status.json", "doc_status")
                ]
                
                for filename, table_name in json_files:
                    file_path = self.lightrag_storage_dir / filename
                    if file_path.exists():
                        result = self.backup_json_file(file_path, table_name)
                        if result["success"] and result["records"] > 0:
                            logger.info(f"📦 Backed up {result['records']} records from {filename}")
                
                # Wait before next check
                time.sleep(self.check_interval)
                
            except Exception as e:
                logger.error(f"❌ Error in backup monitor: {e}")
                time.sleep(self.check_interval)
    
    def start_monitoring(self):
        """Start the backup monitoring in a separate thread"""
        if self.running:
            logger.warning("⚠️ Backup monitor is already running")
            return
        
        self.running = True
        self.monitor_thread = threading.Thread(target=self.monitor_and_backup, daemon=True)
        self.monitor_thread.start()
        logger.info("🚀 LightRAG backup monitor started")
    
    def stop_monitoring(self):
        """Stop the backup monitoring"""
        self.running = False
        if self.monitor_thread:
            self.monitor_thread.join()
        logger.info("🛑 LightRAG backup monitor stopped")
    
    def get_backup_stats(self) -> Dict[str, Any]:
        """Get backup statistics"""
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
            logger.error(f"❌ Failed to get backup stats: {e}")
            return {"error": str(e)}
    
    def query_backup_data(self, query_type: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Query backed up data"""
        try:
            conn = sqlite3.connect(self.backup_db_path)
            cursor = conn.cursor()
            
            if query_type == "documents":
                cursor.execute("""
                    SELECT doc_id, content, last_updated 
                    FROM full_docs 
                    ORDER BY last_updated DESC 
                    LIMIT ?
                """, (limit,))
                columns = ["doc_id", "content", "last_updated"]
                
            elif query_type == "chunks":
                cursor.execute("""
                    SELECT chunk_id, content, doc_id, last_updated 
                    FROM text_chunks 
                    ORDER BY last_updated DESC 
                    LIMIT ?
                """, (limit,))
                columns = ["chunk_id", "content", "doc_id", "last_updated"]
                
            elif query_type == "status":
                cursor.execute("""
                    SELECT doc_id, status, last_updated 
                    FROM doc_status 
                    ORDER BY last_updated DESC 
                    LIMIT ?
                """, (limit,))
                columns = ["doc_id", "status", "last_updated"]
            
            results = []
            for row in cursor.fetchall():
                results.append(dict(zip(columns, row)))
            
            conn.close()
            return results
            
        except Exception as e:
            logger.error(f"❌ Failed to query backup data: {e}")
            return []

def main():
    """Main function to run the backup monitor"""
    print("🔬 LightRAG Backup Monitor")
    print("=" * 50)
    
    monitor = LightRAGBackupMonitor()
    
    try:
        # Start monitoring
        monitor.start_monitoring()
        
        # Show initial stats
        stats = monitor.get_backup_stats()
        print(f"\n📊 Backup Statistics:")
        print(f"   Full Documents: {stats.get('full_docs_count', 0)}")
        print(f"   Text Chunks: {stats.get('text_chunks_count', 0)}")
        print(f"   Document Status: {stats.get('doc_status_count', 0)}")
        print(f"   Last Backup: {stats.get('last_backup', 'Never')}")
        print(f"   Backup DB: {stats.get('backup_db_path', 'N/A')}")
        
        print(f"\n🔄 Monitoring LightRAG files every 30 seconds...")
        print("Press Ctrl+C to stop")
        
        # Keep running
        while True:
            time.sleep(60)
            
    except KeyboardInterrupt:
        print("\n🛑 Stopping backup monitor...")
        monitor.stop_monitoring()
        
        # Show final stats
        stats = monitor.get_backup_stats()
        print(f"\n📊 Final Backup Statistics:")
        print(f"   Full Documents: {stats.get('full_docs_count', 0)}")
        print(f"   Text Chunks: {stats.get('text_chunks_count', 0)}")
        print(f"   Document Status: {stats.get('doc_status_count', 0)}")
        print(f"   Last Backup: {stats.get('last_backup', 'Never')}")

if __name__ == "__main__":
    main() 