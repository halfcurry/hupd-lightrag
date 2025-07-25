#!/usr/bin/env python3
"""
SQLite to PostgreSQL Data Sync Script

This script syncs session data from SQLite to PostgreSQL for Grafana visualization.
"""

import sqlite3
import psycopg2
import json
import logging
from datetime import datetime
from typing import Dict, List, Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import os

from dotenv import load_dotenv

load_dotenv()

POSTGRES_USER = os.getenv("LOCAL_POSTGRES_USER", "myuser")
POSTGRES_PASSWORD = os.getenv("LOCAL_POSTGRES_PASSWORD", "mysecretpassword")
POSTGRES_DB = os.getenv("LOCAL_POSTGRES_DB", "patent_monitoring")

class DataSync:
    def __init__(self, sqlite_path: str = "sessions.db", postgres_config: Dict = None):
        self.sqlite_path = sqlite_path
        self.postgres_config = postgres_config or {
            'host': 'localhost',
            'port': 5432,
            'database': POSTGRES_DB,
            'user': POSTGRES_USER,
            'password': POSTGRES_PASSWORD
        }
    
    def get_sqlite_connection(self):
        """Get SQLite connection"""
        return sqlite3.connect(self.sqlite_path)
    
    def get_postgres_connection(self):
        """Get PostgreSQL connection"""
        return psycopg2.connect(**self.postgres_config)
    
    def sync_session_metadata(self):
        """Sync session metadata from SQLite to PostgreSQL"""
        try:
            sqlite_conn = self.get_sqlite_connection()
            postgres_conn = self.get_postgres_connection()
            
            # Get all session metadata from SQLite
            sqlite_cursor = sqlite_conn.cursor()
            sqlite_cursor.execute("""
                SELECT session_id, start_time, end_time, total_interactions, 
                       evaluation_status, overall_score, detailed_metrics
                FROM session_metadata
            """)
            
            sessions = sqlite_cursor.fetchall()
            logger.info(f"Found {len(sessions)} sessions to sync")
            
            # Insert/update in PostgreSQL
            postgres_cursor = postgres_conn.cursor()
            
            for session in sessions:
                session_id, start_time, end_time, total_interactions, evaluation_status, overall_score, detailed_metrics = session
                
                # Convert detailed_metrics to JSONB
                detailed_metrics_json = None
                if detailed_metrics:
                    try:
                        detailed_metrics_json = json.dumps(json.loads(detailed_metrics))
                    except:
                        detailed_metrics_json = None
                
                # Upsert session metadata
                postgres_cursor.execute("""
                    INSERT INTO session_metadata 
                    (session_id, start_time, end_time, total_interactions, evaluation_status, overall_score, detailed_metrics)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (session_id) 
                    DO UPDATE SET
                        end_time = EXCLUDED.end_time,
                        total_interactions = EXCLUDED.total_interactions,
                        evaluation_status = EXCLUDED.evaluation_status,
                        overall_score = EXCLUDED.overall_score,
                        detailed_metrics = EXCLUDED.detailed_metrics
                """, (session_id, start_time, end_time, total_interactions, evaluation_status, overall_score, detailed_metrics_json))
            
            postgres_conn.commit()
            logger.info(f"Synced {len(sessions)} session metadata records")
            
            sqlite_conn.close()
            postgres_conn.close()
            
        except Exception as e:
            logger.error(f"Error syncing session metadata: {e}")
    
    def sync_session_interactions(self):
        """Sync session interactions from SQLite to PostgreSQL"""
        try:
            sqlite_conn = self.get_sqlite_connection()
            postgres_conn = self.get_postgres_connection()
            
            # Get all interactions from SQLite
            sqlite_cursor = sqlite_conn.cursor()
            sqlite_cursor.execute("""
                SELECT session_id, timestamp, query_text, response_text, data_source,
                       response_time_ms, guardrail_scores, evaluation_scores, interaction_type, session_phase
                FROM session_interactions
            """)
            
            interactions = sqlite_cursor.fetchall()
            logger.info(f"Found {len(interactions)} interactions to sync")
            
            # Insert/update in PostgreSQL
            postgres_cursor = postgres_conn.cursor()
            
            for interaction in interactions:
                session_id, timestamp, query_text, response_text, data_source, response_time_ms, guardrail_scores, evaluation_scores, interaction_type, session_phase = interaction
                
                # Convert JSON strings to JSONB
                guardrail_json = None
                evaluation_json = None
                
                if guardrail_scores:
                    try:
                        guardrail_json = json.dumps(json.loads(guardrail_scores))
                    except:
                        guardrail_json = None
                
                if evaluation_scores:
                    try:
                        evaluation_json = json.dumps(json.loads(evaluation_scores))
                    except:
                        evaluation_json = None
                
                # Check if interaction already exists (by session_id, timestamp, and query_text)
                postgres_cursor.execute("""
                    SELECT id FROM session_interactions 
                    WHERE session_id = %s AND timestamp = %s AND query_text = %s
                """, (session_id, timestamp, query_text))
                
                existing = postgres_cursor.fetchone()
                
                if existing:
                    # Update existing interaction
                    postgres_cursor.execute("""
                        UPDATE session_interactions 
                        SET response_text = %s, data_source = %s, response_time_ms = %s,
                            guardrail_scores = %s, evaluation_scores = %s, interaction_type = %s, session_phase = %s
                        WHERE id = %s
                    """, (response_text, data_source, response_time_ms, guardrail_json, evaluation_json, interaction_type, session_phase, existing[0]))
                else:
                    # Insert new interaction
                    postgres_cursor.execute("""
                        INSERT INTO session_interactions 
                        (session_id, timestamp, query_text, response_text, data_source, response_time_ms, 
                         guardrail_scores, evaluation_scores, interaction_type, session_phase)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """, (session_id, timestamp, query_text, response_text, data_source, response_time_ms, 
                         guardrail_json, evaluation_json, interaction_type, session_phase))
            
            postgres_conn.commit()
            logger.info(f"Synced {len(interactions)} interaction records")
            
            sqlite_conn.close()
            postgres_conn.close()
            
        except Exception as e:
            logger.error(f"Error syncing session interactions: {e}")
    
    def sync_all(self):
        """Sync all data from SQLite to PostgreSQL"""
        logger.info("Starting SQLite to PostgreSQL sync...")
        self.sync_session_metadata()
        self.sync_session_interactions()
        logger.info("Sync completed!")

if __name__ == "__main__":
    sync = DataSync()
    sync.sync_all() 