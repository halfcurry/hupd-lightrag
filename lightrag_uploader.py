#!/usr/bin/env python3
"""
LightRAG Text Ingestion Script with PostgreSQL Status Tracking

This script iterates through a directory of JSON files. For each file, it:
1. Extracts an 'application_number'.
2. Checks its status in a PostgreSQL database.
3. If not already successful, it inserts the file's raw content as text
   to the LightRAG server and updates the status in the database.

Configuration is managed via a .env file.
"""

import os
import sys
import json
import argparse
import hashlib
import logging
import time
from pathlib import Path
from typing import Set, List, Dict, Optional, Tuple
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import psycopg2
from psycopg2.extras import RealDictCursor
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# --- Configuration ---
DB_USER = os.getenv("POSTGRES_USER")
DB_PASSWORD = os.getenv("POSTGRES_PASSWORD")
DB_HOST = os.getenv("POSTGRES_HOST", "localhost")
DB_PORT = os.getenv("POSTGRES_PORT", "5432")
DB_NAME = os.getenv("POSTGRES_DATABASE")
SERVER_URL = os.getenv("LIGHTRAG_SERVER_URL", "http://localhost:9621")
API_KEY = os.getenv("LIGHTRAG_API_KEY")

# Construct PostgreSQL URL
DB_URL = None
if all([DB_USER, DB_PASSWORD, DB_HOST, DB_PORT, DB_NAME]):
    DB_URL = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
else:
    logging.error("Missing one or more required PostgreSQL environment variables. Exiting.")
    sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('lightrag_ingestion.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class PostgreSQLStatusTracker:
    """Handles PostgreSQL database operations for upload status tracking."""
    
    def __init__(self, db_url: str):
        self.db_url = db_url
        self.table_name = "lightrag_upload_status"
        self._init_database()
    
    def _get_connection(self):
        try:
            return psycopg2.connect(self.db_url)
        except psycopg2.Error as e:
            logger.error(f"Database connection error: {e}")
            raise
    
    def _init_database(self):
        """Initializes the database table for upload status tracking."""
        create_table_sql = f"""
        CREATE TABLE IF NOT EXISTS {self.table_name} (
            id SERIAL PRIMARY KEY,
            application_number VARCHAR(255) UNIQUE NOT NULL,
            file_path TEXT NOT NULL,
            file_hash VARCHAR(64) NOT NULL,
            file_size BIGINT NOT NULL,
            upload_status VARCHAR(20) NOT NULL DEFAULT 'pending',
            upload_timestamp TIMESTAMP WITH TIME ZONE,
            created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
            error_message TEXT
        );
        CREATE INDEX IF NOT EXISTS idx_application_number ON {self.table_name}(application_number);
        CREATE INDEX IF NOT EXISTS idx_upload_status ON {self.table_name}(upload_status);
        """
        try:
            with self._get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(create_table_sql)
            logger.info("Database table verified successfully.")
        except psycopg2.Error as e:
            logger.error(f"Error initializing database: {e}")
            raise
    
    def get_file_status(self, application_number: str) -> Optional[Dict]:
        """Gets status of a specific file by application number."""
        try:
            with self._get_connection() as conn:
                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    cur.execute(f"SELECT upload_status FROM {self.table_name} WHERE application_number = %s", (application_number,))
                    return cur.fetchone()
        except psycopg2.Error as e:
            logger.error(f"Error getting file status for {application_number}: {e}")
            return None
    
    def ensure_record_exists(self, application_number: str, file_path: str, file_hash: str, file_size: int):
        """Creates or updates a record, setting its status to 'pending' if it was previously failed."""
        try:
            with self._get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(f"""
                        INSERT INTO {self.table_name} 
                        (application_number, file_path, file_hash, file_size, upload_status)
                        VALUES (%s, %s, %s, %s, 'pending')
                        ON CONFLICT (application_number) DO UPDATE SET
                            file_path = EXCLUDED.file_path,
                            file_hash = EXCLUDED.file_hash,
                            file_size = EXCLUDED.file_size,
                            upload_status = 'pending',
                            updated_at = CURRENT_TIMESTAMP,
                            error_message = NULL
                        WHERE {self.table_name}.upload_status != 'success';
                    """, (application_number, file_path, file_hash, file_size))
        except psycopg2.Error as e:
            logger.error(f"Error ensuring record exists for {application_number}: {e}")
            raise
    
    def mark_as_success(self, application_number: str):
        """Marks a file as successfully processed."""
        try:
            with self._get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(f"""
                        UPDATE {self.table_name} 
                        SET upload_status = 'success', upload_timestamp = CURRENT_TIMESTAMP, updated_at = CURRENT_TIMESTAMP, error_message = NULL
                        WHERE application_number = %s
                    """, (application_number,))
        except psycopg2.Error as e:
            logger.error(f"Error marking {application_number} as success: {e}")
            raise
            
    def mark_as_failed(self, application_number: str, error_message: str):
        """Marks a file as failed to process."""
        try:
            with self._get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(f"""
                        UPDATE {self.table_name} 
                        SET upload_status = 'failed', updated_at = CURRENT_TIMESTAMP, error_message = %s
                        WHERE application_number = %s
                    """, (error_message, application_number))
        except psycopg2.Error as e:
            logger.error(f"Error marking {application_number} as failed: {e}")
            raise

class TextIngestor:
    """Handles the process of ingesting text content to the LightRAG server."""
    
    def __init__(self, server_url: str, api_key: Optional[str], db_tracker: PostgreSQLStatusTracker):
        self.server_url = server_url.rstrip('/')
        self.api_key = api_key
        self.db_tracker = db_tracker
        self.session = self._create_session()

    def _create_session(self) -> requests.Session:
        session = requests.Session()
        retry = Retry(total=3, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504])
        adapter = HTTPAdapter(max_retries=retry)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        if self.api_key:
            session.headers.update({'Authorization': f'Bearer {self.api_key}'})
        return session

    def _get_file_hash(self, file_path: Path) -> str:
        hash_md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()

    def _extract_application_number(self, file_path: Path) -> Optional[str]:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f).get('application_number')
        except (json.JSONDecodeError, IOError, KeyError, TypeError) as e:
            logger.warning(f"Could not extract application_number from {file_path.name}: {e}")
            return None

    def _insert_text_content(self, text: str, description: str) -> Tuple[bool, str]:
        """Inserts text using the API and returns status and response text."""
        payload = {"text": text, "description": description}
        headers = self.session.headers.copy()
        headers['Content-Type'] = 'application/json'
        try:
            response = self.session.post(f"{self.server_url}/documents/text", json=payload, headers=headers, timeout=120)
            if response.status_code == 200:
                return True, response.text
            return False, f"HTTP {response.status_code}: {response.text}"
        except requests.RequestException as e:
            return False, str(e)

    def process_folder(self, directory: Path) -> Dict[str, int]:
        """Main processing loop for the folder."""
        if not directory.is_dir():
            logger.error(f"Provided path is not a directory: {directory}")
            return {'processed': 0, 'skipped': 0, 'failed': 0}

        logger.info(f"Starting to process folder: {directory}")
        json_files = list(directory.rglob('*.json'))
        total_files = len(json_files)
        logger.info(f"Found {total_files} .json files.")

        stats = {'processed': 0, 'skipped': 0, 'failed': 0}

        for i, file_path in enumerate(json_files):
            logger.debug(f"Checking file {i+1}/{total_files}: {file_path.name}")
            app_number = self._extract_application_number(file_path)
            if not app_number:
                stats['failed'] += 1
                continue
            
            # 1. Check status in PostgreSQL
            status = self.db_tracker.get_file_status(app_number)
            if status and status.get('upload_status') == 'success':
                logger.info(f"Skipping '{app_number}' (already processed).")
                stats['skipped'] += 1
                continue

            # 2. Prepare for processing
            logger.info(f"Processing '{app_number}' from file {file_path.name}.")
            try:
                file_hash = self._get_file_hash(file_path)
                file_size = file_path.stat().st_size
                text_content = file_path.read_text(encoding='utf-8')
            except (IOError, FileNotFoundError) as e:
                logger.error(f"Failed to read file for '{app_number}': {e}")
                stats['failed'] += 1
                continue

            # 3. Ensure a record exists in the DB (marks as pending)
            self.db_tracker.ensure_record_exists(app_number, str(file_path), file_hash, file_size)

            # 4. Insert text content via API
            success, response_msg = self._insert_text_content(text_content, f"Content for application {app_number}")
            
            # 5. Update status based on API response
            if success:
                self.db_tracker.mark_as_success(app_number)
                logger.info(f"Successfully processed and ingested '{app_number}'.")
                stats['processed'] += 1
            else:
                self.db_tracker.mark_as_failed(app_number, response_msg)
                logger.error(f"Failed to ingest '{app_number}'. Reason: {response_msg}")
                stats['failed'] += 1
            
            time.sleep(0.1) # Small delay between API calls

        return stats

def main():
    parser = argparse.ArgumentParser(
        description='Ingests JSON file contents as text into a LightRAG instance.',
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        'folder',
        type=str,
        help='Path to the folder containing .json files to process.'
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose (DEBUG) logging.'
    )
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    try:
        db_tracker = PostgreSQLStatusTracker(DB_URL)
        ingestor = TextIngestor(
            server_url=SERVER_URL,
            api_key=API_KEY,
            db_tracker=db_tracker
        )
        
        folder_path = Path(args.folder)
        results = ingestor.process_folder(folder_path)

        logger.info("--- Processing Summary ---")
        logger.info(f"Successfully Processed: {results['processed']}")
        logger.info(f"Skipped (already done): {results['skipped']}")
        logger.info(f"Failed: {results['failed']}")
        logger.info("--------------------------")
        
        sys.exit(1 if results['failed'] > 0 else 0)

    except Exception as e:
        logger.error(f"A fatal error occurred: {e}", exc_info=args.verbose)
        sys.exit(1)

if __name__ == "__main__":
    main()