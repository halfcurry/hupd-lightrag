#!/usr/bin/env python3
"""
LightRAG Backup Management Script

Manage the LightRAG backup monitoring system.
"""

import subprocess
import time
import signal
import sys
import os
from pathlib import Path

class BackupManager:
    """Manage LightRAG backup monitoring"""
    
    def __init__(self):
        self.backup_script = "lightrag_backup_monitor.py"
        self.query_script = "backup_query_tool.py"
        self.backup_db = "lightrag_backup.db"
        self.process = None
    
    def start_backup_monitor(self):
        """Start the backup monitoring process"""
        try:
            if self.is_backup_running():
                print("⚠️  Backup monitor is already running")
                return False
            
            print("🚀 Starting LightRAG backup monitor...")
            self.process = subprocess.Popen(
                [sys.executable, self.backup_script],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            
            # Wait a moment to see if it starts successfully
            time.sleep(2)
            if self.process.poll() is None:
                print("✅ Backup monitor started successfully")
                print(f"   Process ID: {self.process.pid}")
                print("   Monitoring LightRAG files every 30 seconds")
                return True
            else:
                print("❌ Failed to start backup monitor")
                return False
                
        except Exception as e:
            print(f"❌ Error starting backup monitor: {e}")
            return False
    
    def stop_backup_monitor(self):
        """Stop the backup monitoring process"""
        try:
            if self.is_backup_running():
                print("🛑 Stopping backup monitor...")
                subprocess.run(["pkill", "-f", "lightrag_backup_monitor"])
                time.sleep(2)
                print("✅ Backup monitor stopped")
            else:
                print("⚠️  Backup monitor is not running")
                
        except Exception as e:
            print(f"❌ Error stopping backup monitor: {e}")
    
    def is_backup_running(self):
        """Check if backup monitor is running"""
        try:
            result = subprocess.run(
                ["pgrep", "-f", "lightrag_backup_monitor"],
                capture_output=True,
                text=True
            )
            return result.returncode == 0
        except:
            return False
    
    def get_backup_status(self):
        """Get backup system status"""
        status = {
            "monitor_running": self.is_backup_running(),
            "backup_db_exists": Path(self.backup_db).exists(),
            "backup_db_size": 0
        }
        
        if status["backup_db_exists"]:
            status["backup_db_size"] = Path(self.backup_db).stat().st_size
        
        return status
    
    def show_backup_stats(self):
        """Show backup statistics"""
        try:
            result = subprocess.run(
                [sys.executable, self.query_script],
                input="4\n5\n",  # Show stats then exit
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if result.returncode == 0:
                # Extract stats from output
                lines = result.stdout.split('\n')
                for line in lines:
                    if 'Full Documents:' in line or 'Text Chunks:' in line or 'Last Backup:' in line:
                        print(f"   {line.strip()}")
            else:
                print("❌ Failed to get backup stats")
                
        except Exception as e:
            print(f"❌ Error getting backup stats: {e}")
    
    def query_backup_data(self):
        """Open interactive query tool"""
        try:
            print("🔍 Opening backup query tool...")
            subprocess.run([sys.executable, self.query_script])
        except Exception as e:
            print(f"❌ Error opening query tool: {e}")

def main():
    """Main management interface"""
    manager = BackupManager()
    
    while True:
        print("\n🔬 LightRAG Backup Manager")
        print("=" * 40)
        
        # Show current status
        status = manager.get_backup_status()
        print(f"📊 Status:")
        print(f"   Monitor Running: {'✅ YES' if status['monitor_running'] else '❌ NO'}")
        print(f"   Backup DB: {'✅ EXISTS' if status['backup_db_exists'] else '❌ MISSING'}")
        if status['backup_db_exists']:
            print(f"   DB Size: {status['backup_db_size']:,} bytes")
        
        print(f"\nOptions:")
        print("1. Start backup monitor")
        print("2. Stop backup monitor")
        print("3. Show backup statistics")
        print("4. Query backup data")
        print("5. Exit")
        
        choice = input("\nSelect option (1-5): ").strip()
        
        if choice == "1":
            manager.start_backup_monitor()
        
        elif choice == "2":
            manager.stop_backup_monitor()
        
        elif choice == "3":
            print("\n📊 Backup Statistics:")
            manager.show_backup_stats()
        
        elif choice == "4":
            manager.query_backup_data()
        
        elif choice == "5":
            print("👋 Goodbye!")
            break
        
        else:
            print("❌ Invalid option")
        
        time.sleep(1)

if __name__ == "__main__":
    main() 