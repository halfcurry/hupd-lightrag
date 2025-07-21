#!/usr/bin/env python3
"""
Optimized Patent Filter Script for Specific G06 Subcategory Main IPCR Label
Filters patents where main_ipcr_label contains "G06N" or "G06V" and creates highly optimized versions.
These subcategories cover AI/ML (G06N) and computer vision (G06V) technologies.
Using main_ipcr_label provides more precise filtering than ipcr_labels.
Keeps only essential fields and truncates long text to reduce processing time.
"""

import json
import shutil
import logging
from pathlib import Path
from typing import Dict, Any, List
import argparse

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class OptimizedG06PatentFilter:
    def __init__(self, input_dir: str, output_dir: str):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Essential fields to keep
        self.essential_fields = {
            'application_number',
            'publication_number', 
            'title',
            'decision',
            'date_produced',
            'date_published',
            'main_ipcr_label',
            'ipcr_labels',
            'filing_date',
            'patent_number',
            'inventor_list',
            'abstract',
            'summary'
        }
        
        # Text field limits (characters)
        self.text_limits = {
            'title': 500,
            'abstract': 1000,
            'summary': 2000,
            'full_description': 5000  # Only if we keep it
        }
    
    def has_g06_main_ipcr_label(self, patent_data: Dict[str, Any]) -> bool:
        """Check if patent has specific G06 subcategory main_ipcr_label"""
        main_ipcr_label = patent_data.get('main_ipcr_label', '')
        if isinstance(main_ipcr_label, str):
            # Look for specific G06 subcategories: G06N, G06V
            target_labels = ["G06N", "G06V"]
            return any(main_ipcr_label.startswith(target) for target in target_labels)
        return False
    
    def truncate_text(self, text: str, max_length: int) -> str:
        """Truncate text to specified length"""
        if not text or len(text) <= max_length:
            return text
        return text[:max_length] + "..."
    
    def optimize_patent_data(self, patent_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create optimized version of patent data with only essential fields"""
        optimized_data = {}
        
        # Keep only essential fields
        for field in self.essential_fields:
            if field in patent_data:
                value = patent_data[field]
                
                # Truncate text fields if they're too long
                if field in self.text_limits and isinstance(value, str):
                    value = self.truncate_text(value, self.text_limits[field])
                
                optimized_data[field] = value
        
        # Add processing metadata
        optimized_data['_processing_info'] = {
            'original_size': len(json.dumps(patent_data)),
            'optimized_size': len(json.dumps(optimized_data)),
            'reduction_percent': round((1 - len(json.dumps(optimized_data)) / len(json.dumps(patent_data))) * 100, 2)
        }
        
        return optimized_data
    
    def process_patent_file(self, file_path: Path) -> bool:
        """Process a single patent JSON file"""
        try:
            # Read JSON file
            with open(file_path, 'r', encoding='utf-8') as f:
                patent_data = json.load(f)
            
            # Check if patent has G06 main_ipcr_label
            if self.has_g06_main_ipcr_label(patent_data):
                # Create optimized version
                optimized_data = self.optimize_patent_data(patent_data)
                
                # Save optimized version
                output_file = self.output_dir / file_path.name
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(optimized_data, f, ensure_ascii=False, indent=2)
                
                # Log the filtered patent
                title = optimized_data.get('title', 'Unknown')
                application_number = optimized_data.get('application_number', 'Unknown')
                main_ipcr_label = optimized_data.get('main_ipcr_label', 'Unknown')
                reduction = optimized_data.get('_processing_info', {}).get('reduction_percent', 0)
                
                logger.info(f"✅ Filtered: {title} (App: {application_number}, Main IPCR Label: {main_ipcr_label}, Size reduction: {reduction}%)")
                return True
            else:
                logger.debug(f"⏭️ Skipped (no G06N/G06V in main_ipcr_label): {file_path.name}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Error processing {file_path.name}: {e}")
            return False
    
    def process_directory(self) -> bool:
        """Process all JSON files in the input directory and subdirectories"""
        json_files = list(self.input_dir.rglob("*.json"))
        total_files = len(json_files)
        processed_count = 0
        filtered_count = 0
        
        logger.info(f"Starting to process {total_files} files from {self.input_dir} (including subdirectories)")
        
        for i, file_path in enumerate(json_files, 1):
            if self.process_patent_file(file_path):
                filtered_count += 1
            processed_count += 1
            
            # Progress update every 1000 files
            if i % 1000 == 0:
                logger.info(f"Progress: {i}/{total_files} files processed, {filtered_count} G06N/G06V main_ipcr_label patents found")
        
        logger.info(f"✅ Processing complete! Processed {processed_count} files, found {filtered_count} G06N/G06V main_ipcr_label patents")
        return True

def main():
    parser = argparse.ArgumentParser(description="Filter G06 patents with optimized processing")
    parser.add_argument("input_dir", help="Input directory containing patent JSON files")
    parser.add_argument("--output", "-o", default="hupd_processed", help="Output directory for filtered patents")
    
    args = parser.parse_args()
    
    filter = OptimizedG06PatentFilter(args.input_dir, args.output)
    filter.process_directory()

if __name__ == "__main__":
    main() 