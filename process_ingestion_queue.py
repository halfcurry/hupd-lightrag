#!/usr/bin/env python3
"""
Process Ingestion Queue

This script processes the ingestion queue and adds patents to the RAG system.
"""

import json
import os
import sys
from pathlib import Path
import requests
import time

def process_ingestion_queue():
    """Process all files in the ingestion queue"""
    print("🔄 Processing Ingestion Queue")
    print("=" * 40)
    
    ingestion_dir = Path("ingestion_queue")
    if not ingestion_dir.exists():
        print("❌ Ingestion queue directory not found")
        return
    
    # Get all JSON files in the queue
    queue_files = list(ingestion_dir.glob("*.json"))
    
    if not queue_files:
        print("✅ Ingestion queue is empty")
        return
    
    print(f"📋 Found {len(queue_files)} files in ingestion queue")
    
    processed_count = 0
    failed_count = 0
    
    for file_path in queue_files:
        try:
            print(f"\n📄 Processing: {file_path.name}")
            
            # Read patent data
            with open(file_path, 'r', encoding='utf-8') as f:
                patent_data = json.load(f)
            
            # Convert to text format for LightRAG
            text_content = _convert_to_text_format(patent_data)
            
            # Upload to LightRAG
            success = _upload_to_lightrag(text_content, patent_data)
            
            if success:
                # Move file to processed directory
                processed_dir = ingestion_dir / "processed"
                processed_dir.mkdir(exist_ok=True)
                
                processed_file = processed_dir / file_path.name
                file_path.rename(processed_file)
                
                print(f"✅ Successfully processed: {file_path.name}")
                processed_count += 1
            else:
                print(f"❌ Failed to process: {file_path.name}")
                failed_count += 1
                
        except Exception as e:
            print(f"❌ Error processing {file_path.name}: {e}")
            failed_count += 1
    
    print(f"\n📊 Processing Summary:")
    print(f"   ✅ Successfully processed: {processed_count}")
    print(f"   ❌ Failed: {failed_count}")
    print(f"   📋 Total files: {len(queue_files)}")

def _convert_to_text_format(patent_data: dict) -> str:
    """Convert patent data to text format for LightRAG"""
    
    text_content = f"""PATENT ANALYSIS REPORT

Patent Number: {patent_data.get('patent_number', 'Unknown')}
Title: {patent_data.get('title', 'Unknown')}
Status: {patent_data.get('status', 'Unknown')}
Main IPC Code: {patent_data.get('main_ipc_code', 'Unknown')}
Source: {patent_data.get('source', 'Google Patents')}

{f"Inventors: {', '.join(patent_data.get('inventors', []))}" if patent_data.get('inventors') else "Inventors: Not available"}

{f"Assignee: {patent_data.get('assignee')}" if patent_data.get('assignee') else "Assignee: Not available"}

ABSTRACT:
{patent_data.get('abstract', 'No abstract available')}

TECHNICAL DETAILS:
This patent is classified under {patent_data.get('main_ipc_code', 'Unknown')} which indicates {_get_ipc_description(patent_data.get('main_ipc_code', ''))}.

PATENT ANALYSIS:
This patent represents an innovation in the field of {_get_field_from_ipc(patent_data.get('main_ipc_code', ''))}. The technology described in this patent has potential applications in various industries and may represent significant prior art for related inventions.

KEY FEATURES:
- Patent Number: {patent_data.get('patent_number', 'Unknown')}
- Classification: {patent_data.get('main_ipc_code', 'Unknown')}
- Status: {patent_data.get('status', 'Unknown')}
- Source: {patent_data.get('source', 'Google Patents')}
"""
    
    return text_content

def _get_ipc_description(ipc_code: str) -> str:
    """Get description of IPC code"""
    if ipc_code.startswith("G06N"):
        return "Computer systems based on specific computational models (Neural Networks, Machine Learning)"
    elif ipc_code.startswith("G06V"):
        return "Image or video recognition or understanding (Computer Vision, Pattern Recognition)"
    else:
        return "Computing technology"

def _get_field_from_ipc(ipc_code: str) -> str:
    """Get field description from IPC code"""
    if ipc_code.startswith("G06N"):
        return "artificial intelligence and machine learning"
    elif ipc_code.startswith("G06V"):
        return "computer vision and image processing"
    else:
        return "computing technology"

def _upload_to_lightrag(text_content: str, patent_data: dict) -> bool:
    """Upload patent data to LightRAG"""
    try:
        # Prepare payload for LightRAG
        payload = {
            "text": text_content,
            "metadata": {
                "source": "Google Patents",
                "reference": "https://patents.google.com/",
                "citation": f"Patent {patent_data.get('patent_number', 'Unknown')}",
                "license": "Patent data from Google Patents",
                "original_file": f"patent_{patent_data.get('patent_number', 'unknown')}.json",
                "filename": f"patent_{patent_data.get('patent_number', 'unknown')}.json",
                "upload_time": time.strftime("%Y-%m-%d %H:%M:%S"),
                "dataset": "Google Patents",
                "patent_type": "G06 (Computing; Calculating; Counting)",
                "patent_number": patent_data.get('patent_number'),
                "main_ipc_code": patent_data.get('main_ipc_code'),
                "status": patent_data.get('status'),
                "ingestion_priority": patent_data.get('ingestion_priority', 'normal')
            }
        }
        
        # Upload to LightRAG
        response = requests.post(
            "http://localhost:9621/documents/text",
            json=payload,
            timeout=300
        )
        
        if response.status_code == 200:
            print(f"   ✅ Successfully uploaded to LightRAG")
            return True
        else:
            print(f"   ❌ LightRAG upload failed: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"   ❌ Error uploading to LightRAG: {e}")
        return False

if __name__ == "__main__":
    process_ingestion_queue() 