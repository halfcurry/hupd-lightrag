#!/usr/bin/env python3
"""
Session Database Module

This module provides database functionality for storing session-specific conversation history
and evaluation metrics. Each session gets its own table for detailed tracking.
"""

import sqlite3
import json
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any
from pathlib import Path
import threading
import time

logger = logging.getLogger(__name__)

class SessionDatabase:
    """Database manager for session-specific conversation and evaluation data"""
    
    def __init__(self, db_path: str = "sessions.db"):
        self.db_path = db_path
        self.lock = threading.Lock()
        self._init_database()
    
    def _init_database(self):
        """Initialize the session database with required tables"""
        try:
            conn = sqlite3.connect(self.db_path, timeout=60.0)
            # Enable WAL mode for better concurrency
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.execute("PRAGMA cache_size=10000")
            cursor = conn.cursor()
            
            # Create session metadata table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS session_metadata (
                    session_id VARCHAR(100) PRIMARY KEY,
                    start_time TIMESTAMP,
                    end_time TIMESTAMP,
                    total_interactions INTEGER DEFAULT 0,
                    evaluation_status VARCHAR(50) DEFAULT 'pending',
                    overall_score FLOAT DEFAULT 0.0,
                    detailed_metrics TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Create session interactions table (for all sessions)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS session_interactions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id VARCHAR(100),
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    query_text TEXT,
                    response_text TEXT,
                    data_source VARCHAR(50),
                    response_time_ms INTEGER,
                    guardrail_scores TEXT,
                    evaluation_scores TEXT,
                    interaction_type VARCHAR(50) DEFAULT 'llm_rag',
                    session_phase VARCHAR(50) DEFAULT 'active'
                )
            """)
            
            # Create indexes for better performance
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_session_interactions_session_id 
                ON session_interactions(session_id)
            """)
            
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_session_interactions_timestamp 
                ON session_interactions(timestamp)
            """)
            
            conn.commit()
            conn.close()
            logger.info("Session database initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing session database: {e}")
            raise
    
    def _get_connection(self):
        """Get a database connection with proper configuration"""
        conn = sqlite3.connect(self.db_path, timeout=60.0)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        return conn
    
    def create_session(self, session_id: str) -> bool:
        """Create a new session in the database"""
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            # Create session metadata entry
            cursor.execute("""
                INSERT OR REPLACE INTO session_metadata 
                (session_id, start_time, evaluation_status)
                VALUES (?, ?, ?)
            """, (session_id, datetime.now().isoformat(), 'pending'))
            
            conn.commit()
            conn.close()
            logger.info(f"Session {session_id} created successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error creating session {session_id}: {e}")
            return False
    
    def record_interaction(self, session_id: str, query: str, response: str, 
                          data_source: str = "llm_rag", response_time_ms: int = 0,
                          guardrail_scores: Dict = None, evaluation_scores: Dict = None,
                          interaction_type: str = "llm_rag") -> bool:
        """Record an interaction in the session interactions table"""
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            # Prepare data
            guardrail_json = json.dumps(guardrail_scores) if guardrail_scores else None
            evaluation_json = json.dumps(evaluation_scores) if evaluation_scores else None
            
            # Record in session interactions table
            cursor.execute("""
                INSERT INTO session_interactions 
                (session_id, query_text, response_text, data_source, response_time_ms, 
                 guardrail_scores, evaluation_scores, interaction_type)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (session_id, query, response, data_source, response_time_ms,
                 guardrail_json, evaluation_json, interaction_type))
            
            # Update session metadata
            cursor.execute("""
                UPDATE session_metadata 
                SET total_interactions = total_interactions + 1
                WHERE session_id = ?
            """, (session_id,))
            
            conn.commit()
            conn.close()
            return True
            
        except Exception as e:
            logger.error(f"Error recording interaction for session {session_id}: {e}")
            return False
    
    def close_session(self, session_id: str, overall_score: float = 0.0, 
                     detailed_metrics: Dict = None) -> bool:
        """Close a session and calculate final evaluation metrics"""
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            # Update session metadata
            detailed_metrics_json = json.dumps(detailed_metrics) if detailed_metrics else None
            cursor.execute("""
                UPDATE session_metadata 
                SET end_time = ?, evaluation_status = ?, overall_score = ?, detailed_metrics = ?
                WHERE session_id = ?
            """, (datetime.now().isoformat(), 'completed', overall_score, 
                  detailed_metrics_json, session_id))
            
            # Update session interactions to mark as closed
            cursor.execute("""
                UPDATE session_interactions 
                SET session_phase = 'closed'
                WHERE session_id = ? AND session_phase = 'active'
            """, (session_id,))
            
            conn.commit()
            conn.close()
            logger.info(f"Session {session_id} closed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error closing session {session_id}: {e}")
            return False
    
    def get_session_interactions(self, session_id: str) -> List[Dict]:
        """Get all interactions for a specific session"""
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT timestamp, query_text, response_text, data_source, 
                       response_time_ms, guardrail_scores, evaluation_scores, interaction_type
                FROM session_interactions 
                WHERE session_id = ?
                ORDER BY timestamp
            """, (session_id,))
            
            interactions = []
            for row in cursor.fetchall():
                interaction = {
                    'timestamp': row[0],
                    'query_text': row[1],
                    'response_text': row[2],
                    'data_source': row[3],
                    'response_time_ms': row[4],
                    'guardrail_scores': json.loads(row[5]) if row[5] else None,
                    'evaluation_scores': json.loads(row[6]) if row[6] else None,
                    'interaction_type': row[7]
                }
                interactions.append(interaction)
            
            conn.close()
            return interactions
            
        except Exception as e:
            logger.error(f"Error getting session interactions for {session_id}: {e}")
            return []
    
    def calculate_session_evaluation(self, session_id: str) -> Dict:
        """Calculate comprehensive evaluation metrics for a session"""
        try:
            interactions = self.get_session_interactions(session_id)
            
            if not interactions:
                return {
                    'session_id': session_id,
                    'total_interactions': 0,
                    'overall_score': 0.0,
                    'average_scores': {},
                    'evaluation_status': 'no_data'
                }
            
            # Calculate average scores
            total_evaluations = 0
            avg_scores = {
                'relevance_score': 0.0,
                'coherence_score': 0.0,
                'factual_accuracy': 0.0,
                'completeness': 0.0,
                'logical_flow': 0.0,
                'contextual_consistency': 0.0,
                'topical_relevance_unity': 0.0,
                'reference_resolution': 0.0,
                'discourse_structure_cohesion': 0.0,
                'faithfulness_retrieval_chain': 0.0,
                'temporal_causal_coherence': 0.0,
                'semantic_coherence': 0.0
            }
            
            for interaction in interactions:
                if interaction['evaluation_scores']:
                    total_evaluations += 1
                    for key in avg_scores:
                        if key in interaction['evaluation_scores']:
                            avg_scores[key] += interaction['evaluation_scores'][key]
            
            # Calculate averages
            if total_evaluations > 0:
                for key in avg_scores:
                    avg_scores[key] /= total_evaluations
            
            # Calculate overall score
            overall_score = sum(avg_scores.values()) / len(avg_scores) if avg_scores else 0.0
            
            return {
                'session_id': session_id,
                'total_interactions': len(interactions),
                'total_evaluations': total_evaluations,
                'overall_score': overall_score,
                'average_scores': avg_scores,
                'evaluation_status': 'completed'
            }
            
        except Exception as e:
            logger.error(f"Error calculating session evaluation for {session_id}: {e}")
            return {
                'session_id': session_id,
                'total_interactions': 0,
                'overall_score': 0.0,
                'average_scores': {},
                'evaluation_status': 'error'
            }
    
    def get_session_summary(self, session_id: str) -> Dict:
        """Get a comprehensive summary of a session"""
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            # Get session metadata
            cursor.execute("""
                SELECT start_time, end_time, total_interactions, evaluation_status, 
                       overall_score, detailed_metrics
                FROM session_metadata 
                WHERE session_id = ?
            """, (session_id,))
            
            row = cursor.fetchone()
            if not row:
                conn.close()
                return {'error': 'Session not found'}
            
            metadata = {
                'session_id': session_id,
                'start_time': row[0],
                'end_time': row[1],
                'total_interactions': row[2],
                'evaluation_status': row[3],
                'overall_score': row[4],
                'detailed_metrics': json.loads(row[5]) if row[5] else None
            }
            
            conn.close()
            
            # Get interactions
            interactions = self.get_session_interactions(session_id)
            
            # Calculate evaluation
            evaluation = self.calculate_session_evaluation(session_id)
            
            return {
                'metadata': metadata,
                'interactions': interactions,
                'evaluation': evaluation
            }
            
        except Exception as e:
            logger.error(f"Error getting session summary for {session_id}: {e}")
            return {'error': str(e)}

    def detect_session_closure(self, session_id: str, current_time: float = None) -> bool:
        """Detect if a session should be closed based on inactivity or completion criteria"""
        try:
            if current_time is None:
                current_time = time.time()
            
            conn = self._get_connection()
            cursor = conn.cursor()
            
            # Get session metadata
            cursor.execute("""
                SELECT start_time, total_interactions, evaluation_status
                FROM session_metadata 
                WHERE session_id = ?
            """, (session_id,))
            
            row = cursor.fetchone()
            if not row:
                conn.close()
                return False
            
            start_time_str, total_interactions, evaluation_status = row
            
            # If already completed, no need to check
            if evaluation_status == 'completed':
                conn.close()
                return True
            
            # Parse start time
            try:
                start_time = datetime.fromisoformat(start_time_str).timestamp()
            except:
                start_time = current_time - 3600  # Default to 1 hour ago
            
            # Get last interaction time
            cursor.execute("""
                SELECT MAX(timestamp) 
                FROM session_interactions 
                WHERE session_id = ?
            """, (session_id,))
            
            last_interaction_row = cursor.fetchone()
            last_interaction_time = None
            
            if last_interaction_row and last_interaction_row[0]:
                try:
                    last_interaction_time = datetime.fromisoformat(last_interaction_row[0]).timestamp()
                except:
                    last_interaction_time = start_time
            
            conn.close()
            
            # Closure criteria
            session_duration = current_time - start_time
            inactivity_duration = current_time - last_interaction_time if last_interaction_time else session_duration
            
            # Close if:
            # 1. Session is longer than 4 hours
            # 2. No interaction for more than 30 minutes
            # 3. Session has been active for more than 6 hours (absolute max)
            
            should_close = (
                session_duration > 14400 or  # 4 hours
                (inactivity_duration > 1800 and total_interactions > 0) or  # 30 minutes inactivity
                session_duration > 21600  # 6 hours max
            )
            
            if should_close:
                logger.info(f"Session {session_id} should be closed: duration={session_duration:.0f}s, "
                          f"inactivity={inactivity_duration:.0f}s, interactions={total_interactions}")
            
            return should_close
            
        except Exception as e:
            logger.error(f"Error detecting session closure for {session_id}: {e}")
            return False
    
    def get_active_sessions(self) -> List[str]:
        """Get list of active session IDs"""
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT session_id 
                FROM session_metadata 
                WHERE evaluation_status = 'pending'
                ORDER BY start_time DESC
            """)
            
            active_sessions = [row[0] for row in cursor.fetchall()]
            conn.close()
            return active_sessions
            
        except Exception as e:
            logger.error(f"Error getting active sessions: {e}")
            return []
    
    def auto_close_expired_sessions(self) -> List[str]:
        """Automatically close all expired sessions"""
        try:
            active_sessions = self.get_active_sessions()
            closed_sessions = []
            
            for session_id in active_sessions:
                if self.detect_session_closure(session_id):
                    # Calculate final evaluation
                    evaluation = self.calculate_session_evaluation(session_id)
                    overall_score = evaluation.get('overall_score', 0.0)
                    
                    # Close the session
                    if self.close_session(session_id, overall_score, evaluation):
                        closed_sessions.append(session_id)
                        logger.info(f"Auto-closed session: {session_id}")
            
            return closed_sessions
            
        except Exception as e:
            logger.error(f"Error auto-closing expired sessions: {e}")
            return []

    def calculate_post_session_evaluation(self, session_id: str) -> Dict:
        """Calculate comprehensive post-session evaluation metrics"""
        try:
            interactions = self.get_session_interactions(session_id)
            
            if not interactions:
                return {
                    'session_id': session_id,
                    'evaluation_status': 'no_data',
                    'overall_score': 0.0,
                    'detailed_metrics': {},
                    'summary': 'No interactions found for session'
                }
            
            # Calculate comprehensive metrics
            total_interactions = len(interactions)
            evaluated_interactions = [i for i in interactions if i['evaluation_scores']]
            total_evaluations = len(evaluated_interactions)
            
            # Initialize metric accumulators
            metrics = {
                'relevance_score': 0.0,
                'coherence_score': 0.0,
                'factual_accuracy': 0.0,
                'completeness': 0.0,
                'logical_flow': 0.0,
                'contextual_consistency': 0.0,
                'topical_relevance_unity': 0.0,
                'reference_resolution': 0.0,
                'discourse_structure_cohesion': 0.0,
                'faithfulness_retrieval_chain': 0.0,
                'temporal_causal_coherence': 0.0,
                'semantic_coherence': 0.0
            }
            
            # Calculate averages for evaluated interactions
            if total_evaluations > 0:
                for interaction in evaluated_interactions:
                    eval_scores = interaction['evaluation_scores']
                    for key in metrics:
                        if key in eval_scores:
                            metrics[key] += eval_scores[key]
                
                # Calculate averages
                for key in metrics:
                    metrics[key] /= total_evaluations
            
            # Calculate overall score
            overall_score = sum(metrics.values()) / len(metrics) if metrics else 0.0
            
            # Calculate additional session-level metrics
            session_metrics = {
                'total_interactions': total_interactions,
                'evaluated_interactions': total_evaluations,
                'evaluation_coverage': total_evaluations / total_interactions if total_interactions > 0 else 0.0,
                'average_response_time': sum(i.get('response_time_ms', 0) for i in interactions) / total_interactions if total_interactions > 0 else 0.0,
                'data_source_distribution': {},
                'interaction_type_distribution': {}
            }
            
            # Calculate data source distribution
            data_sources = {}
            interaction_types = {}
            for interaction in interactions:
                data_source = interaction.get('data_source', 'unknown')
                interaction_type = interaction.get('interaction_type', 'unknown')
                
                data_sources[data_source] = data_sources.get(data_source, 0) + 1
                interaction_types[interaction_type] = interaction_types.get(interaction_type, 0) + 1
            
            session_metrics['data_source_distribution'] = data_sources
            session_metrics['interaction_type_distribution'] = interaction_types
            
            # Calculate quality metrics
            quality_metrics = {
                'high_quality_responses': 0,  # responses with overall score > 0.7
                'medium_quality_responses': 0,  # responses with overall score 0.4-0.7
                'low_quality_responses': 0,  # responses with overall score < 0.4
                'response_length_stats': {
                    'min_length': min(len(i.get('response_text', '')) for i in interactions),
                    'max_length': max(len(i.get('response_text', '')) for i in interactions),
                    'avg_length': sum(len(i.get('response_text', '')) for i in interactions) / total_interactions if total_interactions > 0 else 0
                }
            }
            
            # Categorize response quality
            for interaction in evaluated_interactions:
                eval_scores = interaction['evaluation_scores']
                if eval_scores:
                    response_score = sum(eval_scores.values()) / len(eval_scores)
                    if response_score > 0.7:
                        quality_metrics['high_quality_responses'] += 1
                    elif response_score > 0.4:
                        quality_metrics['medium_quality_responses'] += 1
                    else:
                        quality_metrics['low_quality_responses'] += 1
            
            # Generate comprehensive evaluation summary
            evaluation_summary = {
                'session_id': session_id,
                'evaluation_status': 'completed',
                'overall_score': overall_score,
                'detailed_metrics': metrics,
                'session_metrics': session_metrics,
                'quality_metrics': quality_metrics,
                'summary': f"Session completed with {total_interactions} interactions, "
                          f"{total_evaluations} evaluated. Overall score: {overall_score:.3f}",
                'timestamp': datetime.now().isoformat()
            }
            
            return evaluation_summary
            
        except Exception as e:
            logger.error(f"Error calculating post-session evaluation for {session_id}: {e}")
            return {
                'session_id': session_id,
                'evaluation_status': 'error',
                'overall_score': 0.0,
                'detailed_metrics': {},
                'summary': f"Error calculating evaluation: {str(e)}"
            }
    
    def close_session_with_evaluation(self, session_id: str) -> Dict:
        """Close a session and return comprehensive evaluation results"""
        try:
            # Calculate post-session evaluation
            evaluation = self.calculate_post_session_evaluation(session_id)
            overall_score = evaluation.get('overall_score', 0.0)
            
            # Close the session
            success = self.close_session(session_id, overall_score, evaluation)
            
            if success:
                logger.info(f"Session {session_id} closed with evaluation: {overall_score:.3f}")
                return evaluation
            else:
                logger.error(f"Failed to close session {session_id}")
                return {'error': 'Failed to close session'}
                
        except Exception as e:
            logger.error(f"Error closing session with evaluation {session_id}: {e}")
            return {'error': str(e)}

# Global instance
session_db = SessionDatabase() 