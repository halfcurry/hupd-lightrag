#!/usr/bin/env python3
"""
LightRAG Storage Synchronization

This module provides functionality to automatically synchronize LightRAG storage
between rag_storage (backup) and lightrag_storage (active) directories.
"""

import os
import shutil
import json
import logging
import hashlib
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import threading
from dataclasses import dataclass
import watchdog
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

logger = logging.getLogger(__name__)

@dataclass
class FileInfo:
    """File information for synchronization"""
    path: str
    size: int
    modified_time: float
    hash: str
    exists: bool = True

class LightRAGFileWatcher(FileSystemEventHandler):
    """File system watcher for LightRAG storage changes"""
    
    def __init__(self, sync_manager):
        self.sync_manager = sync_manager
        self.last_sync_time = 0
        self.sync_cooldown = 5  # Minimum seconds between syncs
        
    def on_created(self, event):
        """Handle file creation events"""
        if not event.is_directory:
            self._trigger_sync("file_created", event.src_path)
    
    def on_modified(self, event):
        """Handle file modification events"""
        if not event.is_directory:
            self._trigger_sync("file_modified", event.src_path)
    
    def on_moved(self, event):
        """Handle file move/rename events"""
        if not event.is_directory:
            self._trigger_sync("file_moved", event.src_path)
    
    def _trigger_sync(self, event_type: str, file_path: str):
        """Trigger sync with cooldown to prevent excessive syncing"""
        current_time = time.time()
        
        # Check cooldown to prevent excessive syncing
        if current_time - self.last_sync_time < self.sync_cooldown:
            logger.debug(f"Sync skipped due to cooldown: {event_type} - {file_path}")
            return
        
        self.last_sync_time = current_time
        logger.info(f"File system event detected: {event_type} - {file_path}")
        
        # Trigger sync in a separate thread to avoid blocking
        sync_thread = threading.Thread(
            target=self.sync_manager._triggered_sync,
            args=(event_type, file_path),
            daemon=True
        )
        sync_thread.start()

class LightRAGStorageSync:
    """Synchronizes LightRAG storage between rag_storage and lightrag_storage"""
    
    def __init__(self, 
                 source_dir: str = "rag_storage",
                 target_dir: str = "lightrag_storage",
                 sync_interval: int = 60,
                 enable_watcher: bool = True):
        """
        Initialize LightRAG storage sync
        
        Args:
            source_dir: Source directory (rag_storage)
            target_dir: Target directory (lightrag_storage)
            sync_interval: Sync interval in seconds (fallback)
            enable_watcher: Enable file system watcher
        """
        # Get the project root directory (parent of monitoring directory)
        current_dir = Path(__file__).parent
        project_root = current_dir.parent
        
        self.source_dir = project_root / source_dir
        self.target_dir = project_root / target_dir
        self.sync_interval = sync_interval
        self.sync_running = False
        self.sync_thread = None
        self.enable_watcher = enable_watcher
        
        # File system watcher
        self.observer = None
        self.file_watcher = None
        
        # Ensure directories exist
        self.source_dir.mkdir(exist_ok=True)
        self.target_dir.mkdir(exist_ok=True)
        
        # Track sync history
        self.sync_history = []
        self.last_sync_time = None
        
        logger.info(f"LightRAG Storage Sync initialized: {source_dir} → {target_dir}")
        logger.info(f"Source path: {self.source_dir}")
        logger.info(f"Target path: {self.target_dir}")
        logger.info(f"File watcher enabled: {enable_watcher}")
    
    def start_auto_sync(self) -> bool:
        """Start automatic synchronization"""
        if self.sync_running:
            logger.warning("Auto-sync already running")
            return False
        
        self.sync_running = True
        
        if self.enable_watcher:
            # Start file system watcher
            self._start_file_watcher()
            logger.info("LightRAG file watcher started")
        else:
            # Start polling-based sync
            self.sync_thread = threading.Thread(target=self._auto_sync_worker, daemon=True)
            self.sync_thread.start()
            logger.info("LightRAG polling-based auto-sync started")
        
        return True
    
    def stop_auto_sync(self) -> bool:
        """Stop automatic synchronization"""
        if not self.sync_running:
            logger.warning("Auto-sync not running")
            return False
        
        self.sync_running = False
        
        # Stop file watcher
        if self.observer:
            self.observer.stop()
            self.observer.join(timeout=10)
            self.observer = None
            logger.info("LightRAG file watcher stopped")
        
        # Stop polling thread
        if self.sync_thread:
            self.sync_thread.join(timeout=10)
            self.sync_thread = None
            logger.info("LightRAG polling-based auto-sync stopped")
        
        return True
    
    def _start_file_watcher(self):
        """Start file system watcher"""
        try:
            self.file_watcher = LightRAGFileWatcher(self)
            self.observer = Observer()
            self.observer.schedule(self.file_watcher, str(self.source_dir), recursive=True)
            self.observer.start()
            logger.info(f"File watcher started for: {self.source_dir}")
        except Exception as e:
            logger.error(f"Failed to start file watcher: {e}")
            # Fallback to polling-based sync
            self.sync_thread = threading.Thread(target=self._auto_sync_worker, daemon=True)
            self.sync_thread.start()
            logger.info("Fallback to polling-based auto-sync")
    
    def _triggered_sync(self, event_type: str, file_path: str):
        """Sync triggered by file system event"""
        try:
            logger.info(f"Triggered sync due to {event_type}: {file_path}")
            
            # Perform sync
            result = self.sync_all_files()
            
            # Add event information to result
            result['triggered_by'] = event_type
            result['triggered_file'] = file_path
            
            logger.info(f"Triggered sync completed: {result['files_synced']} synced, "
                       f"{result['files_skipped']} skipped, {result['files_failed']} failed")
            
        except Exception as e:
            logger.error(f"Triggered sync failed: {e}")
    
    def _auto_sync_worker(self):
        """Worker thread for polling-based auto-sync (fallback)"""
        while self.sync_running:
            try:
                self.sync_all_files()
                logger.debug(f"Polling-based auto-sync completed. Next sync in {self.sync_interval}s...")
            except Exception as e:
                logger.error(f"Polling-based auto-sync failed: {e}")
            
            time.sleep(self.sync_interval)
    
    def sync_all_files(self) -> Dict:
        """Synchronize all files from source to target"""
        try:
            sync_results = {
                'timestamp': datetime.now().isoformat(),
                'files_synced': 0,
                'files_skipped': 0,
                'files_failed': 0,
                'total_size_synced': 0,
                'errors': []
            }
            
            # Get file lists
            source_files = self._get_file_list(self.source_dir)
            target_files = self._get_file_list(self.target_dir)
            
            # Sync each file
            for file_info in source_files:
                try:
                    result = self._sync_file(file_info, target_files)
                    if result['synced']:
                        sync_results['files_synced'] += 1
                        sync_results['total_size_synced'] += result['size']
                    elif result['skipped']:
                        sync_results['files_skipped'] += 1
                    else:
                        sync_results['files_failed'] += 1
                        sync_results['errors'].append(result['error'])
                except Exception as e:
                    sync_results['files_failed'] += 1
                    sync_results['errors'].append(str(e))
                    logger.error(f"Failed to sync {file_info.path}: {e}")
            
            # Update sync history
            self.sync_history.append(sync_results)
            self.last_sync_time = datetime.now()
            
            # Keep only last 100 sync records
            if len(self.sync_history) > 100:
                self.sync_history = self.sync_history[-100:]
            
            logger.info(f"LightRAG sync completed: {sync_results['files_synced']} synced, "
                       f"{sync_results['files_skipped']} skipped, {sync_results['files_failed']} failed")
            
            return sync_results
            
        except Exception as e:
            logger.error(f"LightRAG sync failed: {e}")
            return {'error': str(e)}
    
    def _get_file_list(self, directory: Path) -> List[FileInfo]:
        """Get list of files with metadata"""
        files = []
        
        if not directory.exists():
            return files
        
        for file_path in directory.rglob("*"):
            if file_path.is_file():
                try:
                    stat = file_path.stat()
                    file_hash = self._calculate_file_hash(file_path)
                    
                    files.append(FileInfo(
                        path=str(file_path.relative_to(directory)),
                        size=stat.st_size,
                        modified_time=stat.st_mtime,
                        hash=file_hash
                    ))
                except Exception as e:
                    logger.error(f"Error reading file {file_path}: {e}")
        
        return files
    
    def _calculate_file_hash(self, file_path: Path) -> str:
        """Calculate SHA256 hash of file"""
        try:
            hash_sha256 = hashlib.sha256()
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_sha256.update(chunk)
            return hash_sha256.hexdigest()
        except Exception as e:
            logger.error(f"Error calculating hash for {file_path}: {e}")
            return ""
    
    def _sync_file(self, source_file: FileInfo, target_files: List[FileInfo]) -> Dict:
        """Sync a single file from source to target"""
        source_path = self.source_dir / source_file.path
        target_path = self.target_dir / source_file.path
        
        # Ensure target directory exists
        target_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Check if target file exists and compare
        target_file = next((f for f in target_files if f.path == source_file.path), None)
        
        if target_file:
            # File exists in target, check if sync needed
            if (target_file.size == source_file.size and 
                target_file.hash == source_file.hash):
                return {
                    'synced': False,
                    'skipped': True,
                    'size': 0,
                    'reason': 'File identical'
                }
        
        try:
            # Copy file
            shutil.copy2(source_path, target_path)
            
            # Verify copy
            if target_path.exists():
                target_stat = target_path.stat()
                if target_stat.st_size == source_file.size:
                    return {
                        'synced': True,
                        'skipped': False,
                        'size': source_file.size,
                        'reason': 'File synced successfully'
                    }
                else:
                    return {
                        'synced': False,
                        'skipped': False,
                        'size': 0,
                        'error': 'File size mismatch after copy'
                    }
            else:
                return {
                    'synced': False,
                    'skipped': False,
                    'size': 0,
                    'error': 'Target file not created'
                }
                
        except Exception as e:
            return {
                'synced': False,
                'skipped': False,
                'size': 0,
                'error': str(e)
            }
    
    def manual_sync(self) -> Dict:
        """Perform manual synchronization"""
        logger.info("Starting manual LightRAG sync...")
        return self.sync_all_files()
    
    def get_sync_status(self) -> Dict:
        """Get current sync status"""
        return {
            'auto_sync_running': self.sync_running,
            'last_sync_time': self.last_sync_time.isoformat() if self.last_sync_time else None,
            'sync_interval': self.sync_interval,
            'source_dir': str(self.source_dir),
            'target_dir': str(self.target_dir),
            'recent_syncs': self.sync_history[-5:] if self.sync_history else []
        }
    
    def get_storage_stats(self) -> Dict:
        """Get storage statistics"""
        try:
            source_stats = self._get_directory_stats(self.source_dir)
            target_stats = self._get_directory_stats(self.target_dir)
            
            return {
                'source': source_stats,
                'target': target_stats,
                'sync_status': self.get_sync_status()
            }
        except Exception as e:
            logger.error(f"Error getting storage stats: {e}")
            return {'error': str(e)}
    
    def _get_directory_stats(self, directory: Path) -> Dict:
        """Get directory statistics"""
        if not directory.exists():
            return {'exists': False, 'file_count': 0, 'total_size': 0}
        
        total_size = 0
        file_count = 0
        
        for file_path in directory.rglob("*"):
            if file_path.is_file():
                try:
                    total_size += file_path.stat().st_size
                    file_count += 1
                except Exception:
                    pass
        
        return {
            'exists': True,
            'file_count': file_count,
            'total_size': total_size,
            'total_size_mb': round(total_size / (1024 * 1024), 2)
        }
    
    def verify_sync(self) -> Dict:
        """Verify that source and target are in sync"""
        try:
            source_files = self._get_file_list(self.source_dir)
            target_files = self._get_file_list(self.target_dir)
            
            source_hashes = {f.path: f.hash for f in source_files}
            target_hashes = {f.path: f.hash for f in target_files}
            
            missing_files = []
            mismatched_files = []
            
            # Check for missing or mismatched files
            for path, source_hash in source_hashes.items():
                if path not in target_hashes:
                    missing_files.append(path)
                elif target_hashes[path] != source_hash:
                    mismatched_files.append(path)
            
            return {
                'in_sync': len(missing_files) == 0 and len(mismatched_files) == 0,
                'missing_files': missing_files,
                'mismatched_files': mismatched_files,
                'source_file_count': len(source_files),
                'target_file_count': len(target_files)
            }
            
        except Exception as e:
            logger.error(f"Error verifying sync: {e}")
            return {'error': str(e)}

# Global instance for easy access
lightrag_sync = LightRAGStorageSync()

def start_lightrag_auto_sync():
    """Start LightRAG auto-sync"""
    return lightrag_sync.start_auto_sync()

def stop_lightrag_auto_sync():
    """Stop LightRAG auto-sync"""
    return lightrag_sync.stop_auto_sync()

def manual_lightrag_sync():
    """Perform manual LightRAG sync"""
    return lightrag_sync.manual_sync()

def get_lightrag_sync_status():
    """Get LightRAG sync status"""
    return lightrag_sync.get_sync_status()

def get_lightrag_storage_stats():
    """Get LightRAG storage statistics"""
    return lightrag_sync.get_storage_stats()

def verify_lightrag_sync():
    """Verify LightRAG sync status"""
    return lightrag_sync.verify_sync()

if __name__ == "__main__":
    # Test the sync functionality
    import argparse
    
    parser = argparse.ArgumentParser(description="LightRAG Storage Sync")
    parser.add_argument("--action", choices=["sync", "status", "stats", "verify", "start", "stop"], 
                       default="sync", help="Action to perform")
    parser.add_argument("--interval", type=int, default=60, help="Sync interval in seconds")
    
    args = parser.parse_args()
    
    if args.action == "sync":
        result = manual_lightrag_sync()
        print(f"Sync result: {result}")
    elif args.action == "status":
        status = get_lightrag_sync_status()
        print(f"Sync status: {status}")
    elif args.action == "stats":
        stats = get_lightrag_storage_stats()
        print(f"Storage stats: {stats}")
    elif args.action == "verify":
        verify = verify_lightrag_sync()
        print(f"Sync verification: {verify}")
    elif args.action == "start":
        success = start_lightrag_auto_sync()
        print(f"Auto-sync started: {success}")
    elif args.action == "stop":
        success = stop_lightrag_auto_sync()
        print(f"Auto-sync stopped: {success}") 