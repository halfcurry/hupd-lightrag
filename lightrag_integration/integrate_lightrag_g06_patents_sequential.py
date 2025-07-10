#!/usr/bin/env python3
"""
Sequential LightRAG G06 Patent Integration Script
Processes G06 patents one at a time to avoid overwhelming the server
"""

import json
import logging
import signal
import sys
import time
from pathlib import Path
from typing import Dict, List, Set
import requests

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuration
LIGHTRAG_BASE_URL = "http://localhost:9621"
INPUT_DIR = Path("hupd_processed")  # Source directory (safe from deletion)
UPLOAD_DIR = Path("lightrag_upload")  # Directory for LightRAG to consume
MAX_WAIT_TIME = 300  # 5 minutes
POLL_INTERVAL = 2  # 2 seconds

class LightRAGSequentialProcessor:
    def __init__(self):
        self.session = requests.Session()
        self.running = True
        
        # Set up signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        # Create upload directory
        UPLOAD_DIR.mkdir(exist_ok=True)
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        logger.info("Received shutdown signal, stopping gracefully...")
        self.running = False
    
    def check_server_health(self) -> bool:
        """Check if LightRAG server is healthy"""
        try:
            response = self.session.get(f"{LIGHTRAG_BASE_URL}/health", timeout=10)
            if response.status_code == 200:
                data = response.json()
                if data.get("status") == "healthy":
                    logger.info("LightRAG server is healthy")
                    return True
                else:
                    logger.error(f"LightRAG server is not healthy: {data}")
                    return False
            else:
                logger.error(f"LightRAG server health check failed: {response.status_code}")
                return False
        except Exception as e:
            logger.error(f"Error checking LightRAG server health: {e}")
            return False
    
    def clear_all_documents(self) -> bool:
        """Clear all existing documents from LightRAG"""
        try:
            logger.info("Clearing all existing documents...")
            response = self.session.delete(f"{LIGHTRAG_BASE_URL}/documents", timeout=30)
            if response.status_code == 200:
                logger.info("Successfully cleared all documents")
                return True
            else:
                logger.error(f"Failed to clear documents: {response.status_code}")
                return False
        except Exception as e:
            logger.error(f"Error clearing documents: {e}")
            return False
    
    def copy_file_to_upload_dir(self, file_path: Path) -> Path:
        """Copy a file from input directory to upload directory"""
        upload_file = UPLOAD_DIR / file_path.name
        import shutil
        shutil.copy2(file_path, upload_file)
        logger.info(f"Copied {file_path.name} to upload directory")
        return upload_file
    
    def upload_document(self, file_path: Path) -> str | None:
        """Upload a single document to LightRAG and return the document ID if successful"""
        try:
            logger.info(f"Uploading document: {file_path.name}")
            
            # Copy file to upload directory first
            upload_file = self.copy_file_to_upload_dir(file_path)
            
            # Read the JSON file
            with open(upload_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Convert to text format for LightRAG
            text_content = self._json_to_text(data)
            
            # Upload as text document
            payload = {
                "text": text_content,
                "metadata": {
                    "source": "Harvard USPTO Dataset (HUPD)",
                    "reference": "https://patentdataset.org/",
                    "citation": "Suzgun, M., et al. (2022). The Harvard USPTO Patent Dataset: A Large-Scale, Well-Structured, and Multi-Purpose Corpus of Patent Applications. arXiv preprint arXiv:2207.04043",
                    "license": "CreativeCommons Attribution-NonCommercial-ShareAlike 4.0 International",
                    "original_file": str(file_path),
                    "filename": file_path.name,
                    "upload_time": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "dataset": "HUPD",
                    "patent_type": "G06 (Computing; Calculating; Counting)"
                }
            }
            
            # Upload document
            response = self.session.post(
                f"{LIGHTRAG_BASE_URL}/documents/text",
                json=payload,
                timeout=300  # Increased to 5 minutes for large documents
            )
            
            if response.status_code == 200:
                logger.info(f"Successfully uploaded {file_path.name}")
                
                # Try to get document ID from response
                try:
                    response_data = response.json()
                    doc_id = response_data.get("id")
                    if doc_id:
                        return doc_id
                except:
                    pass
                
                # If no ID in response, try to find it in status list
                logger.warning(f"Upload succeeded but no document ID returned for {file_path.name}. Attempting to find document ID in status list.")
                time.sleep(2)  # Wait for document to appear in status list
                
                try:
                    status_response = self.session.get(f"{LIGHTRAG_BASE_URL}/documents", timeout=10)
                    if status_response.status_code == 200:
                        data = status_response.json()
                        
                        # Look for document in processed and processing statuses
                        processed_docs = data.get("statuses", {}).get("processed", [])
                        processing_docs = data.get("statuses", {}).get("processing", [])
                        
                        logger.info(f"Found {len(processed_docs)} processed documents and {len(processing_docs)} processing documents")
                        
                        # Try to find by filename in metadata first
                        for doc in processed_docs + processing_docs:
                            if isinstance(doc, dict) and "metadata" in doc:
                                metadata = doc.get("metadata", {})
                                if metadata.get("filename") == file_path.name:
                                    doc_id = doc.get("id")
                                    status = "processed" if doc in processed_docs else "processing"
                                    logger.info(f"Found document ID for {file_path.name} by metadata: {doc_id} (status: {status})")
                                    return doc_id
                        
                        # If not found by metadata, try to find by content summary
                        title_start = f"Title: {data.get('title', '')[:50]}"
                        logger.info(f"Looking for document with title starting with: {title_start}")
                        
                        for doc in processed_docs + processing_docs:
                            if isinstance(doc, dict) and "content_summary" in doc:
                                content_summary = doc.get("content_summary", "")
                                if content_summary.startswith(title_start):
                                    doc_id = doc.get("id")
                                    status = "processed" if doc in processed_docs else "processing"
                                    logger.info(f"Found document ID for {file_path.name} by content match: {doc_id} (status: {status})")
                                    return doc_id
                        
                        # If still not found, try to find the most recent document (LightRAG might not preserve metadata)
                        if processed_docs:
                            # Get the most recent processed document
                            latest_doc = max(processed_docs, key=lambda x: x.get("created_at", ""))
                            doc_id = latest_doc.get("id")
                            logger.info(f"Using most recent processed document ID for {file_path.name}: {doc_id}")
                            return doc_id
                        elif processing_docs:
                            # Get the most recent processing document
                            latest_doc = max(processing_docs, key=lambda x: x.get("created_at", ""))
                            doc_id = latest_doc.get("id")
                            logger.info(f"Using most recent processing document ID for {file_path.name}: {doc_id}")
                            return doc_id
                        
                        logger.error(f"Could not find document ID for {file_path.name} in status list")
                        return None
                    else:
                        logger.error(f"Failed to get document status: {status_response.status_code}")
                        return None
                except Exception as e:
                    logger.error(f"Error finding document ID: {e}")
                    return None
            else:
                logger.error(f"Failed to upload {file_path.name}: {response.status_code} - {response.text}")
                return None
        except Exception as e:
            logger.error(f"Error uploading {file_path.name}: {e}")
            return None
    
    def _json_to_text(self, data: Dict) -> str:
        """Convert patent JSON data to text format with proper source attribution"""
        text_parts = []
        
        # Add source attribution header
        text_parts.append("SOURCE: Harvard USPTO Dataset (HUPD)")
        text_parts.append("REFERENCE: https://patentdataset.org/")
        text_parts.append("CITATION: Suzgun, M., et al. (2022). The Harvard USPTO Patent Dataset: A Large-Scale, Well-Structured, and Multi-Purpose Corpus of Patent Applications. arXiv preprint arXiv:2207.04043")
        text_parts.append("LICENSE: CreativeCommons Attribution-NonCommercial-ShareAlike 4.0 International")
        text_parts.append("=" * 80)
        
        # Add basic patent information
        if "title" in data:
            text_parts.append(f"Title: {data['title']}")
        
        if "application_number" in data:
            text_parts.append(f"Application Number: {data['application_number']}")
        
        if "publication_number" in data:
            text_parts.append(f"Publication Number: {data['publication_number']}")
        
        if "decision" in data:
            text_parts.append(f"Decision: {data['decision']}")
        
        # Add IPC labels if available
        if "ipcr_labels" in data and data["ipcr_labels"]:
            text_parts.append(f"IPC Labels: {', '.join(data['ipcr_labels'])}")
        
        # Add abstract if available
        if "abstract" in data and data["abstract"]:
            text_parts.append(f"Abstract: {data['abstract']}")
        
        # Add claims if available
        if "claims" in data and data["claims"]:
            claims_text = "\n".join([f"Claim {i+1}: {claim}" for i, claim in enumerate(data["claims"])])
            text_parts.append(f"Claims:\n{claims_text}")
        
        # Add description if available
        if "description" in data and data["description"]:
            text_parts.append(f"Description: {data['description']}")
        
        return "\n\n".join(text_parts)
    
    def wait_for_document_completion(self, doc_id: str) -> bool:
        """Wait for a document to complete processing (look for 'processed' status)"""
        logger.info(f"Waiting for document {doc_id} to complete processing...")
        start_time = time.time()
        while self.running and (time.time() - start_time) < MAX_WAIT_TIME:
            try:
                response = self.session.get(f"{LIGHTRAG_BASE_URL}/documents", timeout=10)
                if response.status_code == 200:
                    data = response.json()
                    # Check if document is processed
                    processed_docs = data.get("statuses", {}).get("processed", [])
                    for doc in processed_docs:
                        if doc.get("id") == doc_id:
                            logger.info(f"Document {doc_id} processed successfully")
                            return True
                    # Check if document is still processing
                    processing_docs = data.get("statuses", {}).get("processing", [])
                    for doc in processing_docs:
                        if doc.get("id") == doc_id:
                            logger.info(f"Document {doc_id} still processing...")
                            break
                    else:
                        # Document not found in either processed or processing
                        logger.warning(f"Document {doc_id} not found in processed or processing status")
                        return False
                else:
                    logger.error(f"Failed to get document status: {response.status_code}")
                    return False
            except Exception as e:
                logger.error(f"Error checking document status: {e}")
                return False
            time.sleep(POLL_INTERVAL)
        if not self.running:
            logger.error("Process interrupted by user")
            return False
        logger.error(f"Document {doc_id} processing timed out after {MAX_WAIT_TIME} seconds")
        return False
    
    def get_document_status(self) -> tuple[set, set]:
        """Get sets of document filenames that have failed or completed in LightRAG"""
        try:
            response = self.session.get(f"{LIGHTRAG_BASE_URL}/documents", timeout=10)
            if response.status_code == 200:
                data = response.json()
                failed_docs = data.get("statuses", {}).get("failed", {})
                processed_docs = data.get("statuses", {}).get("processed", [])
                
                # Extract filenames from failed documents
                failed_filenames = set()
                for doc_id, doc_info in failed_docs.items():
                    if isinstance(doc_info, dict) and "metadata" in doc_info:
                        metadata = doc_info.get("metadata", {})
                        filename = metadata.get("filename")
                        if filename:
                            failed_filenames.add(filename)
                
                # Extract filenames from processed documents
                processed_filenames = set()
                for doc in processed_docs:
                    if isinstance(doc, dict) and "metadata" in doc:
                        metadata = doc.get("metadata", {})
                        filename = metadata.get("filename")
                        if filename:
                            processed_filenames.add(filename)
                
                logger.info(f"Found {len(failed_filenames)} previously failed documents and {len(processed_filenames)} processed documents")
                return failed_filenames, processed_filenames
            else:
                logger.warning(f"Failed to get document status: {response.status_code}")
                return set(), set()
        except Exception as e:
            logger.warning(f"Error getting document status: {e}")
            return set(), set()
    
    def get_failed_documents(self) -> set:
        """Get set of document filenames that have failed in LightRAG"""
        failed_filenames, _ = self.get_document_status()
        return failed_filenames
    
    def get_pending_documents(self) -> List[Path]:
        """Get list of JSON files to process, excluding only previously completed ones"""
        if not INPUT_DIR.exists():
            logger.error(f"Input directory {INPUT_DIR} does not exist")
            return []
        
        # Get list of all JSON files from source directory
        all_json_files = list(INPUT_DIR.glob("*.json"))
        
        # Get previously failed and processed documents
        failed_filenames, processed_filenames = self.get_document_status()
        
        # Filter out only previously processed documents (allow retrying failed ones)
        pending_files = [f for f in all_json_files if f.name not in processed_filenames]
        
        if failed_filenames:
            logger.info(f"Found {len(failed_filenames)} previously failed documents that will be retried: {list(failed_filenames)[:5]}{'...' if len(failed_filenames) > 5 else ''}")
        
        if processed_filenames:
            logger.info(f"Skipping {len(processed_filenames)} previously processed documents: {list(processed_filenames)[:5]}{'...' if len(processed_filenames) > 5 else ''}")
        
        logger.info(f"Found {len(pending_files)} documents to process (including {len(failed_filenames)} failed documents to retry, excluding {len(processed_filenames)} processed)")
        return pending_files
    
    def process_documents_sequentially(self) -> bool:
        """Process all documents one at a time"""
        # Check server health
        if not self.check_server_health():
            logger.error("LightRAG server is not healthy. Exiting.")
            return False
        # Note: Not clearing documents to preserve existing completed/failed status
        logger.info("Preserving existing document status (completed/failed documents will be skipped/retried)")
        # Get list of documents to process
        documents = self.get_pending_documents()
        if not documents:
            logger.error("No documents found to process")
            return False
        logger.info(f"Starting sequential processing of {len(documents)} documents")
        successful_count = 0
        failed_count = 0
        
        for i, doc_path in enumerate(documents, 1):
            if not self.running:
                logger.info("Process interrupted by user")
                break
                
            logger.info(f"Processing document {i}/{len(documents)}: {doc_path.name}")
            
            # Upload document and get doc_id
            doc_id = self.upload_document(doc_path)
            if not doc_id:
                logger.error(f"Failed to upload {doc_path.name}. Continuing with next document.")
                failed_count += 1
                continue
            
            # Wait for completion
            if not self.wait_for_document_completion(doc_id):
                logger.error(f"Document {doc_path.name} failed to process. Continuing with next document.")
                failed_count += 1
                continue
            
            logger.info(f"Successfully processed {doc_path.name} ({i}/{len(documents)})")
            successful_count += 1
        
        logger.info(f"Processing completed: {successful_count} successful, {failed_count} failed")
        
        if successful_count > 0:
            logger.info("Some documents were processed successfully!")
            return True
        else:
            logger.error("No documents were processed successfully")
            return False

def main():
    """Main function"""
    logger.info("Starting Sequential LightRAG G06 Patent Integration")
    
    processor = LightRAGSequentialProcessor()
    
    try:
        success = processor.process_documents_sequentially()
        if success:
            logger.info("Sequential processing completed successfully")
            sys.exit(0)
        else:
            logger.error("Sequential processing failed")
            sys.exit(1)
    except KeyboardInterrupt:
        logger.info("Process interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 