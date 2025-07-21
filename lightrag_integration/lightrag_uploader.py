"""
LightRAG Uploader Module

This module provides a unified interface for uploading documents to LightRAG.
It wraps the existing LightRAGSequentialProcessor to provide the interface expected by main.py.
"""

import logging
from pathlib import Path
from typing import Dict, Any, List

from .integrate_lightrag_g06_patents_sequential import LightRAGSequentialProcessor

logger = logging.getLogger(__name__)

class LightRAGUploader:
    """
    Unified LightRAG upload interface
    
    This class provides a simplified interface for uploading documents to LightRAG,
    wrapping the existing LightRAGSequentialProcessor functionality.
    """
    
    def __init__(self, lightrag_url: str = "http://localhost:9621"):
        self.lightrag_url = lightrag_url
        self.logger = logging.getLogger(__name__)
        self.processor = None
    
    def upload_documents_sequential(self, source_dir: str = "hupd_processed"):
        """
        Upload documents to LightRAG using sequential processing
        
        Args:
            source_dir: Directory containing documents to upload
        """
        self.logger.info(f"Starting sequential document upload from {source_dir}")
        
        try:
            # Create the processor
            self.processor = LightRAGSequentialProcessor()
            
            # Check server health first
            if not self.processor.check_server_health():
                raise Exception("LightRAG server is not healthy")
            
            # Process documents sequentially
            success = self.processor.process_documents_sequentially()
            
            if success:
                self.logger.info("✅ Document upload completed successfully")
            else:
                self.logger.error("❌ Document upload failed")
                
        except Exception as e:
            self.logger.error(f"❌ Error during document upload: {e}")
            raise
    
    def get_upload_status(self) -> Dict[str, Any]:
        """
        Get the current status of document uploads
        
        Returns:
            Dictionary with upload status information
        """
        try:
            if not self.processor:
                return {"error": "No upload session active"}
            
            # Get document status
            processed, processing = self.processor.get_document_status()
            failed = self.processor.get_failed_documents()
            pending = self.processor.get_pending_documents()
            
            return {
                "processed_count": len(processed),
                "processing_count": len(processing),
                "failed_count": len(failed),
                "pending_count": len(pending),
                "total_documents": len(processed) + len(processing) + len(failed) + len(pending),
                "processed_documents": list(processed),
                "processing_documents": list(processing),
                "failed_documents": list(failed),
                "pending_documents": [str(p) for p in pending]
            }
            
        except Exception as e:
            self.logger.error(f"Error getting upload status: {e}")
            return {"error": str(e)}
    
    def check_server_health(self) -> bool:
        """
        Check if LightRAG server is healthy
        
        Returns:
            True if server is healthy, False otherwise
        """
        try:
            if not self.processor:
                self.processor = LightRAGSequentialProcessor()
            
            return self.processor.check_server_health()
            
        except Exception as e:
            self.logger.error(f"Error checking server health: {e}")
            return False
    
    def clear_documents(self) -> bool:
        """
        Clear all documents from LightRAG
        
        Returns:
            True if successful, False otherwise
        """
        try:
            if not self.processor:
                self.processor = LightRAGSequentialProcessor()
            
            return self.processor.clear_all_documents()
            
        except Exception as e:
            self.logger.error(f"Error clearing documents: {e}")
            return False
    
    def get_upload_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the upload process
        
        Returns:
            Dictionary with upload statistics
        """
        try:
            status = self.get_upload_status()
            
            if "error" in status:
                return status
            
            # Calculate success rate
            total = status["total_documents"]
            processed = status["processed_count"]
            failed = status["failed_count"]
            
            success_rate = (processed / total * 100) if total > 0 else 0
            
            return {
                **status,
                "success_rate_percent": round(success_rate, 2),
                "completion_percent": round((processed + failed) / total * 100, 2) if total > 0 else 0
            }
            
        except Exception as e:
            self.logger.error(f"Error getting upload stats: {e}")
            return {"error": str(e)} 