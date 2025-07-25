#!/usr/bin/env python3
"""
Enhanced Patent Analyzer

This module orchestrates the complete patent analysis workflow:
1. Fetch patent details from Google Patents
2. Analyze with local LLM
3. Search for similar patents in RAG
4. Conditionally ingest into RAG if G06N/G06V
"""

import json
import logging
import time
from typing import Dict, List, Optional, Any
from pathlib import Path
import requests

from .google_patents_api import get_patent_details, search_google_patents

import os

OLLAMA_HOST = os.environ.get("OLLAMA_BINDING_HOST", "")
LIGHTRAG_SERVER = os.environ.get("LIGHTRAG_SERVER_URL", "http://localhost:9621")

logger = logging.getLogger(__name__)

class EnhancedPatentAnalyzer:
    """
    Enhanced patent analyzer with Google Patents integration and RAG workflow
    """
    
    def __init__(self, lightrag_url: str = LIGHTRAG_SERVER, ollama_url: str = OLLAMA_HOST):
        self.lightrag_url = lightrag_url
        self.ollama_url = ollama_url
        self.ingestion_queue_dir = Path("ingestion_queue")
        self.ingestion_queue_dir.mkdir(exist_ok=True)
    
    def analyze_patent_comprehensive(self, patent_query: str) -> Dict[str, Any]:
        """
        Comprehensive patent analysis workflow
        
        Args:
            patent_query: Patent number, title, or search terms
            
        Returns:
            Dictionary with analysis results
        """
        try:
            logger.info(f"Starting comprehensive analysis for: {patent_query}")
            
            # Step 1: Fetch patent details from Google Patents
            patent_data = self._fetch_patent_details(patent_query)
            if not patent_data:
                return {"error": f"No patent found for query: {patent_query}"}
            
            # Step 2: Analyze with local LLM
            llm_analysis = self._analyze_with_llm(patent_data)
            
            # Step 3: Search for similar patents in RAG
            similar_patents = self._search_similar_in_rag(patent_data)
            
            # Step 4: Check if should be ingested into RAG
            should_ingest = self._should_ingest_to_rag(patent_data)
            ingestion_status = None
            
            if should_ingest:
                ingestion_status = self._queue_for_rag_ingestion(patent_data)
            
            # Compile results
            results = {
                "patent_data": patent_data,
                "llm_analysis": llm_analysis,
                "similar_patents": similar_patents,
                "should_ingest": should_ingest,
                "ingestion_status": ingestion_status
            }
            
            logger.info(f"Comprehensive analysis completed for: {patent_query}")
            return results
            
        except Exception as e:
            logger.error(f"Error in comprehensive analysis: {e}")
            return {"error": f"Analysis failed: {str(e)}"}
    
    def _fetch_patent_details(self, patent_query: str) -> Optional[Dict]:
        """Fetch patent details from Google Patents"""
        try:
            # First try to get specific patent details
            patent_data = get_patent_details(patent_query)
            
            if patent_data:
                logger.info(f"Found specific patent: {patent_data['patent_number']}")
                return patent_data
            
            # If not found, try search
            search_results = search_google_patents(patent_query, max_results=1, use_selenium=True)
            if search_results:
                logger.info(f"Found patent via search: {search_results[0]['patent_number']}")
                return search_results[0]
            
            return None
            
        except Exception as e:
            logger.error(f"Error fetching patent details: {e}")
            return None
    
    def _analyze_with_llm(self, patent_data: Dict) -> Dict:
        """Analyze patent data with local LLM"""
        try:
            # Prepare minimal data for LLM analysis
            analysis_data = {
                "patent_number": patent_data.get("patent_number", "Unknown"),
                "title": patent_data.get("title", "Unknown"),
                "abstract": patent_data.get("abstract", "No abstract available"),
                "status": patent_data.get("status", "Unknown"),
                "main_ipc_code": patent_data.get("main_ipc_code", "Unknown"),
                "inventors": patent_data.get("inventors", []),
                "assignee": patent_data.get("assignee", "Unknown")
            }
            
            # Create prompt for LLM analysis
            prompt = f"""Analyze the following patent information and provide insights:

Patent Number: {analysis_data['patent_number']}
Title: {analysis_data['title']}
Abstract: {analysis_data['abstract']}
Status: {analysis_data['status']}
Main IPC Code: {analysis_data['main_ipc_code']}
Inventors: {', '.join(analysis_data['inventors']) if analysis_data['inventors'] else 'Unknown'}
Assignee: {analysis_data['assignee']}

Please provide analysis on:
1. Technical novelty and innovation
2. Patent scope and claims analysis
3. Market potential and commercial implications
4. Prior art considerations
5. Recommendations for patent strategy

Provide a comprehensive analysis:"""
            
            # Generate response using Ollama
            response = requests.post(
                f"{self.ollama_url}/api/generate",
                json={
                    "model": "qwen2.5:14b-instruct",
                    "prompt": prompt,
                    "stream": False
                },
                timeout=180
            )
            
            if response.status_code == 200:
                data = response.json()
                if 'response' in data:
                    return {
                        "analysis": data['response'].strip(),
                        "status": "success"
                    }
            
            return {
                "analysis": "Unable to generate LLM analysis at this time.",
                "status": "error"
            }
            
        except Exception as e:
            logger.error(f"Error in LLM analysis: {e}")
            return {
                "analysis": f"LLM analysis failed: {str(e)}",
                "status": "error"
            }
    
    def _search_similar_in_rag(self, patent_data: Dict) -> List[Dict]:
        """Search for similar patents in RAG system"""
        try:
            # Create query for RAG based on patent data
            rag_query = f"""Find patents similar to:
Title: {patent_data.get('title', '')}
Abstract: {patent_data.get('abstract', '')}
Main IPC Code: {patent_data.get('main_ipc_code', '')}

Return the most relevant similar patents from the database."""
            
            # Query RAG system
            payload = {
                "model": "qwen2.5:14b-instruct",
                "messages": [{"role": "user", "content": rag_query}],
                "stream": False
            }
            
            response = requests.post(
                f"{self.lightrag_url}/api/chat",
                json=payload,
                timeout=120
            )
            
            if response.status_code == 200:
                data = response.json()
                if 'message' in data and 'content' in data['message']:
                    return {
                        "similar_patents": data['message']['content'],
                        "status": "success"
                    }
            
            return {
                "similar_patents": "No similar patents found in RAG database.",
                "status": "error"
            }
            
        except Exception as e:
            logger.error(f"Error searching RAG: {e}")
            return {
                "similar_patents": f"RAG search failed: {str(e)}",
                "status": "error"
            }
    
    def _should_ingest_to_rag(self, patent_data: Dict) -> bool:
        """Check if patent should be ingested into RAG (G06N or G06V)"""
        main_ipc_code = patent_data.get("main_ipc_code", "")
        
        if not main_ipc_code:
            return False
        
        # Check if it's G06N (AI/ML) or G06V (Computer Vision)
        if main_ipc_code.startswith("G06N") or main_ipc_code.startswith("G06V"):
            logger.info(f"Patent {patent_data.get('patent_number')} qualifies for RAG ingestion (IPC: {main_ipc_code})")
            return True
        
        return False
    
    def _queue_for_rag_ingestion(self, patent_data: Dict) -> Dict:
        """Queue patent for RAG ingestion by creating JSON file and attempt immediate upload (hybrid approach)"""
        import shutil
        try:
            # Create JSON file for ingestion
            patent_number = patent_data.get("patent_number", "unknown")
            timestamp = int(time.time())
            filename = f"patent_{patent_number}_{timestamp}.json"
            filepath = self.ingestion_queue_dir / filename
            
            # Prepare data for ingestion (minimal format)
            ingestion_data = {
                "patent_number": patent_data.get("patent_number"),
                "title": patent_data.get("title"),
                "abstract": patent_data.get("abstract"),
                "main_ipc_code": patent_data.get("main_ipc_code"),
                "status": patent_data.get("status"),
                "inventors": patent_data.get("inventors", []),
                "assignee": patent_data.get("assignee"),
                "source": "Google Patents",
                "ingestion_timestamp": timestamp,
                "ingestion_priority": "high" if patent_data.get("main_ipc_code", "").startswith(("G06N", "G06V")) else "normal"
            }
            
            # Write to JSON file
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(ingestion_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Queued patent {patent_number} for RAG ingestion: {filepath}")
            
            # Immediately attempt to upload to LightRAG
            upload_success = self._upload_to_lightrag(ingestion_data)
            if upload_success:
                # Move file to processed/
                processed_dir = self.ingestion_queue_dir / "processed"
                processed_dir.mkdir(exist_ok=True)
                processed_file = processed_dir / filename
                shutil.move(str(filepath), str(processed_file))
                logger.info(f"Patent {patent_number} uploaded and moved to processed: {processed_file}")
                return {
                    "status": "uploaded",
                    "filepath": str(processed_file),
                    "message": f"Patent {patent_number} uploaded to LightRAG and moved to processed."
                }
            else:
                logger.warning(f"Patent {patent_number} could not be uploaded, left in queue for retry.")
                return {
                    "status": "queued",
                    "filepath": str(filepath),
                    "message": f"Patent {patent_number} queued for RAG ingestion (upload failed, will retry later)"
                }
        except Exception as e:
            logger.error(f"Error queuing for RAG ingestion: {e}")
            return {
                "status": "error",
                "message": f"Failed to queue for ingestion: {str(e)}"
            }

    def _upload_to_lightrag(self, patent_data: dict) -> bool:
        """Upload patent data to LightRAG (same logic as process_ingestion_queue.py)"""
        try:
            # Convert to text format
            text_content = self._convert_to_text_format(patent_data)
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
            response = requests.post(
                f"{self.lightrag_url}/documents/text",
                json=payload,
                timeout=300
            )
            if response.status_code == 200:
                logger.info(f"Successfully uploaded patent {patent_data.get('patent_number')} to LightRAG.")
                return True
            else:
                logger.warning(f"LightRAG upload failed: {response.status_code}")
                return False
        except Exception as e:
            logger.error(f"Error uploading to LightRAG: {e}")
            return False

    def _convert_to_text_format(self, patent_data: dict) -> str:
        """Convert patent data to text format for LightRAG"""
        # Build inventors and assignee strings separately to avoid nested f-string issues
        inventors_str = f"Inventors: {', '.join(patent_data.get('inventors', []))}" if patent_data.get('inventors') else 'Inventors: Not available'
        assignee_str = f"Assignee: {patent_data.get('assignee')}" if patent_data.get('assignee') else 'Assignee: Not available'
        
        text_content = f"""PATENT ANALYSIS REPORT\n\nPatent Number: {patent_data.get('patent_number', 'Unknown')}\nTitle: {patent_data.get('title', 'Unknown')}\nStatus: {patent_data.get('status', 'Unknown')}\nMain IPC Code: {patent_data.get('main_ipc_code', 'Unknown')}\nSource: {patent_data.get('source', 'Google Patents')}\n\n{inventors_str}\n\n{assignee_str}\n\nABSTRACT:\n{patent_data.get('abstract', 'No abstract available')}\n\nTECHNICAL DETAILS:\nThis patent is classified under {patent_data.get('main_ipc_code', 'Unknown')} which indicates {self._get_ipc_description(patent_data.get('main_ipc_code', ''))}.\n\nPATENT ANALYSIS:\nThis patent represents an innovation in the field of {self._get_field_from_ipc(patent_data.get('main_ipc_code', ''))}. The technology described in this patent has potential applications in various industries and may represent significant prior art for related inventions.\n\nKEY FEATURES:\n- Patent Number: {patent_data.get('patent_number', 'Unknown')}\n- Classification: {patent_data.get('main_ipc_code', 'Unknown')}\n- Status: {patent_data.get('status', 'Unknown')}\n- Source: {patent_data.get('source', 'Google Patents')}\n"""
        return text_content

    def _get_ipc_description(self, ipc_code: str) -> str:
        if ipc_code.startswith("G06N"):
            return "Computer systems based on specific computational models (Neural Networks, Machine Learning)"
        elif ipc_code.startswith("G06V"):
            return "Image or video recognition or understanding (Computer Vision, Pattern Recognition)"
        else:
            return "Computing technology"

    def _get_field_from_ipc(self, ipc_code: str) -> str:
        if ipc_code.startswith("G06N"):
            return "artificial intelligence and machine learning"
        elif ipc_code.startswith("G06V"):
            return "computer vision and image processing"
        else:
            return "computing technology"
    
    def get_ingestion_queue_status(self) -> Dict:
        """Get status of ingestion queue"""
        try:
            queue_files = list(self.ingestion_queue_dir.glob("*.json"))
            
            return {
                "queue_size": len(queue_files),
                "files": [f.name for f in queue_files],
                "status": "active" if queue_files else "empty"
            }
            
        except Exception as e:
            logger.error(f"Error getting ingestion queue status: {e}")
            return {
                "queue_size": 0,
                "files": [],
                "status": "error"
            } 