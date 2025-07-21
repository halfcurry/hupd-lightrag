#!/usr/bin/env python3
"""
Session Logger for Patent Chatbot

This module logs all session data including:
- Conversation history
- Guardrails scores
- Response metrics
- Evaluation scores
- System performance data
"""

import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
import logging

# Import monitoring system
try:
    from monitoring.postgres_monitor import postgres_monitor
    MONITORING_AVAILABLE = True
except ImportError:
    MONITORING_AVAILABLE = False
    print("⚠️ Monitoring system not available")

# Import session database
try:
    from monitoring.session_database import session_db
    SESSION_DB_AVAILABLE = True
except ImportError:
    SESSION_DB_AVAILABLE = False
    print("⚠️ Session database not available")

logger = logging.getLogger(__name__)

@dataclass
class SessionMetrics:
    """Container for session metrics"""
    session_id: str
    start_time: str
    end_time: Optional[str] = None
    total_queries: int = 0
    total_responses: int = 0
    avg_response_time: float = 0.0
    guardrails_enabled: bool = True
    monitoring_enabled: bool = True
    
    # Guardrails summary
    avg_profanity_score: float = 0.0
    avg_topic_relevance_score: float = 0.0
    avg_politeness_score: float = 0.0
    
    # Evaluation summary
    avg_relevance_score: float = 0.0
    avg_coherence_score: float = 0.0
    
    # Enhanced evaluation metrics
    avg_factual_accuracy: float = 0.0
    avg_completeness: float = 0.0
    avg_logical_flow: float = 0.0
    avg_contextual_consistency: float = 0.0
    avg_topical_relevance_unity: float = 0.0
    avg_reference_resolution: float = 0.0
    avg_discourse_structure_cohesion: float = 0.0
    avg_faithfulness_retrieval_chain: float = 0.0
    avg_temporal_causal_coherence: float = 0.0
    avg_semantic_coherence: float = 0.0
    
    # System health
    lightrag_available: bool = True
    neo4j_fallback_used: int = 0
    sqlite_backup_used: int = 0
    errors_encountered: int = 0

@dataclass
class ConversationEntry:
    """Container for a single conversation entry"""
    timestamp: str
    user_query: str
    assistant_response: str
    response_time: float
    guardrail_scores: Dict[str, float]
    evaluation_scores: Optional[Dict[str, float]] = None
    data_source: str = "lightrag"  # lightrag, neo4j, sqlite
    error_message: Optional[str] = None

class SessionLogger:
    """Logs all session data to JSON files"""
    
    def __init__(self, session_id: str = None):
        self.session_id = session_id or f"session_{int(time.time())}"
        self.start_time = datetime.now().isoformat()
        self.conversation_history: List[ConversationEntry] = []
        self.metrics = SessionMetrics(
            session_id=self.session_id,
            start_time=self.start_time
        )
        
        # Create sessions directory
        self.sessions_dir = Path("sessions")
        self.sessions_dir.mkdir(exist_ok=True)
        
        # Create session in database if available
        if SESSION_DB_AVAILABLE:
            try:
                session_db.create_session(self.session_id)
                logger.info(f"Session created in database: {self.session_id}")
            except Exception as e:
                logger.error(f"Failed to create session in database: {e}")
        
        logger.info(f"Session logger initialized: {self.session_id}")
    
    def log_conversation(self, 
                        user_query: str, 
                        assistant_response: str,
                        response_time: float,
                        guardrail_scores: Dict[str, float],
                        evaluation_scores: Optional[Dict[str, float]] = None,
                        data_source: str = "lightrag",
                        error_message: Optional[str] = None):
        """Log a single conversation entry"""
        
        entry = ConversationEntry(
            timestamp=datetime.now().isoformat(),
            user_query=user_query,
            assistant_response=assistant_response,
            response_time=response_time,
            guardrail_scores=guardrail_scores,
            evaluation_scores=evaluation_scores,
            data_source=data_source,
            error_message=error_message
        )
        
        self.conversation_history.append(entry)
        
        # Update metrics
        self.metrics.total_queries += 1
        self.metrics.total_responses += 1
        
        # Update guardrails averages
        if guardrail_scores:
            self._update_guardrail_averages(guardrail_scores)
        
        # Update evaluation averages
        if evaluation_scores:
            self._update_evaluation_averages(evaluation_scores)
        
        # Track data source usage
        if data_source == "neo4j":
            self.metrics.neo4j_fallback_used += 1
        elif data_source == "sqlite":
            self.metrics.sqlite_backup_used += 1
        
        if error_message:
            self.metrics.errors_encountered += 1
        
        # Record to monitoring system if available
        if MONITORING_AVAILABLE:
            try:
                # Estimate tokens (rough approximation)
                tokens_used = len(user_query.split()) + len(assistant_response.split())
                
                # Record chat metric to monitoring system
                postgres_monitor.record_chat_metric(
                    query_text=user_query,
                    response_text=assistant_response,
                    response_time_ms=int(response_time * 1000),  # Convert to milliseconds
                    tokens_used=tokens_used,
                    model_name="patent_chatbot",
                    source_count=0,  # Will be updated if sources are available
                    guardrail_scores=guardrail_scores,
                    evaluation_scores=evaluation_scores,
                    session_id=self.session_id,
                    user_id="web_user"  # Default for web interface
                )
                
                # Record performance metric
                postgres_monitor.record_performance_metric(
                    component="chatbot",
                    operation="get_response",
                    duration_ms=int(response_time * 1000),
                    success=error_message is None,
                    error_message=error_message,
                    additional_data={
                        "data_source": data_source,
                        "session_id": self.session_id,
                        "guardrail_scores": guardrail_scores,
                        "evaluation_scores": evaluation_scores
                    }
                )
                
            except Exception as e:
                logger.error(f"Failed to record metrics to monitoring system: {e}")
        
        # Record to session database if available
        if SESSION_DB_AVAILABLE:
            try:
                # Record interaction to session database
                session_db.record_interaction(
                    session_id=self.session_id,
                    query=user_query,
                    response=assistant_response,
                    data_source=data_source,
                    response_time_ms=int(response_time * 1000),
                    guardrail_scores=guardrail_scores,
                    evaluation_scores=evaluation_scores,
                    interaction_type="llm_rag" if evaluation_scores else "menu_selection"
                )
                
                # Check for session closure
                session_db.detect_session_closure(self.session_id)
                
            except Exception as e:
                logger.error(f"Failed to record to session database: {e}")
        
        logger.debug(f"Logged conversation entry: {len(self.conversation_history)} total")
    
    def log_query(self, query: str):
        """Log a user query (for compatibility with chatbot)"""
        # This method is called by the chatbot but we don't need to do anything here
        # The actual logging happens in log_conversation when we have the full response
        logger.debug(f"Query logged: {query[:50]}...")
    
    def log_response(self, response: str, response_time: float):
        """Log a chatbot response (for compatibility with chatbot)"""
        # This method is called by the chatbot but we don't need to do anything here
        # The actual logging happens in log_conversation when we have the full response
        logger.debug(f"Response logged: {response[:50]}... (time: {response_time:.2f}s)")
    
    def _update_guardrail_averages(self, scores: Dict[str, float]):
        """Update running averages for guardrail scores"""
        current_count = self.metrics.total_responses
        
        # Update profanity score
        self.metrics.avg_profanity_score = (
            (self.metrics.avg_profanity_score * (current_count - 1) + scores.get('profanity_score', 0.0)) / current_count
        )
        
        # Update topic relevance score
        self.metrics.avg_topic_relevance_score = (
            (self.metrics.avg_topic_relevance_score * (current_count - 1) + scores.get('topic_relevance_score', 0.0)) / current_count
        )
        
        # Update politeness score
        self.metrics.avg_politeness_score = (
            (self.metrics.avg_politeness_score * (current_count - 1) + scores.get('politeness_score', 0.0)) / current_count
        )
    
    def _update_evaluation_averages(self, scores: Dict[str, float]):
        """Update running averages for evaluation scores"""
        current_count = self.metrics.total_responses
        
        # Update basic evaluation scores
        self.metrics.avg_relevance_score = (
            (self.metrics.avg_relevance_score * (current_count - 1) + scores.get('relevance_score', 0.0)) / current_count
        )
        self.metrics.avg_coherence_score = (
            (self.metrics.avg_coherence_score * (current_count - 1) + scores.get('coherence_score', 0.0)) / current_count
        )
        
        # Update enhanced evaluation metrics
        enhanced_metrics = [
            'factual_accuracy', 'completeness', 'logical_flow', 'contextual_consistency',
            'topical_relevance_unity', 'reference_resolution', 'discourse_structure_cohesion',
            'faithfulness_retrieval_chain', 'temporal_causal_coherence', 'semantic_coherence'
        ]
        
        for metric in enhanced_metrics:
            current_avg = getattr(self.metrics, f'avg_{metric}', 0.0)
            new_value = scores.get(metric, 0.0)
            setattr(self.metrics, f'avg_{metric}', 
                   (current_avg * (current_count - 1) + new_value) / current_count)
    
    def update_system_status(self, lightrag_available: bool = True):
        """Update system availability status"""
        self.metrics.lightrag_available = lightrag_available
    
    def save_session(self):
        """Save session data to JSON files"""
        self.metrics.end_time = datetime.now().isoformat()
        
        # Create session directory
        session_dir = self.sessions_dir / self.session_id
        session_dir.mkdir(exist_ok=True)
        
        # Save conversation history
        conversation_file = session_dir / "conversation.json"
        conversation_data = {
            "session_id": self.session_id,
            "start_time": self.start_time,
            "end_time": self.metrics.end_time,
            "conversation": [asdict(entry) for entry in self.conversation_history]
        }
        
        with open(conversation_file, 'w') as f:
            json.dump(conversation_data, f, indent=2, default=str)
        
        # Save metrics summary
        metrics_file = session_dir / "metrics.json"
        metrics_data = asdict(self.metrics)
        # Round float values to 3 decimal places
        metrics_data = self._round_float_values(metrics_data, 3)
        with open(metrics_file, 'w') as f:
            json.dump(metrics_data, f, indent=2, default=str)
        
        # Save detailed metrics
        detailed_metrics = {
            "session_id": self.session_id,
            "start_time": self.start_time,
            "end_time": self.metrics.end_time,
            "metrics": asdict(self.metrics),
            "conversation_summary": {
                "total_entries": len(self.conversation_history),
                "error_summary": self._get_error_summary()
            }
        }
        
        # Round float values to 3 decimal places
        detailed_metrics = self._round_float_values(detailed_metrics, 3)
        
        detailed_file = session_dir / "detailed_metrics.json"
        with open(detailed_file, 'w') as f:
            json.dump(detailed_metrics, f, indent=2, default=str)
        
        # Close session in database if available
        if SESSION_DB_AVAILABLE:
            try:
                session_db.close_session_with_evaluation(self.session_id)
                logger.info(f"Session closed in database: {self.session_id}")
            except Exception as e:
                logger.error(f"Failed to close session in database: {e}")
        
        logger.info(f"Session saved to: {session_dir}")
        return session_dir
    
    def _get_data_source_breakdown(self) -> Dict[str, int]:
        """Get breakdown of data sources used"""
        breakdown = {"lightrag": 0, "neo4j": 0, "sqlite": 0}
        for entry in self.conversation_history:
            if entry.data_source in breakdown:
                breakdown[entry.data_source] += 1
            else:
                # Handle unknown data sources
                breakdown[entry.data_source] = breakdown.get(entry.data_source, 0) + 1
        return breakdown
    
    def _get_error_summary(self) -> List[str]:
        """Get list of error messages"""
        errors = []
        for entry in self.conversation_history:
            if entry.error_message:
                errors.append(entry.error_message)
        return errors
    
    def _round_float_values(self, obj, decimal_places=3):
        """Recursively round float values in a dictionary or object to specified decimal places"""
        if isinstance(obj, dict):
            return {k: self._round_float_values(v, decimal_places) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._round_float_values(item, decimal_places) for item in obj]
        elif isinstance(obj, float):
            return round(obj, decimal_places)
        else:
            return obj
    
    def get_session_summary(self) -> Dict[str, Any]:
        """Get a summary of the current session"""
        summary = {
            "session_id": self.session_id,
            "duration": self._calculate_duration(),
            "total_queries": self.metrics.total_queries,
            "avg_response_time": self.metrics.avg_response_time,
            "guardrails_summary": {
                "avg_profanity": self.metrics.avg_profanity_score,
                "avg_topic_relevance": self.metrics.avg_topic_relevance_score,
                "avg_politeness": self.metrics.avg_politeness_score
            },
            "evaluation_summary": {
                "avg_relevance": self.metrics.avg_relevance_score,
                "avg_coherence": self.metrics.avg_coherence_score,
                "avg_factual_accuracy": self.metrics.avg_factual_accuracy,
                "avg_completeness": self.metrics.avg_completeness,
                "avg_logical_flow": self.metrics.avg_logical_flow,
                "avg_contextual_consistency": self.metrics.avg_contextual_consistency,
                "avg_topical_relevance_unity": self.metrics.avg_topical_relevance_unity,
                "avg_reference_resolution": self.metrics.avg_reference_resolution,
                "avg_discourse_structure_cohesion": self.metrics.avg_discourse_structure_cohesion,
                "avg_faithfulness_retrieval_chain": self.metrics.avg_faithfulness_retrieval_chain,
                "avg_temporal_causal_coherence": self.metrics.avg_temporal_causal_coherence,
                "avg_semantic_coherence": self.metrics.avg_semantic_coherence
            },
            "system_usage": {
                "lightrag_available": self.metrics.lightrag_available,
                "neo4j_fallback_used": self.metrics.neo4j_fallback_used,
                "sqlite_backup_used": self.metrics.sqlite_backup_used,
                "errors_encountered": self.metrics.errors_encountered
            }
        }
        
        # Round all float values to 3 decimal places
        return self._round_float_values(summary, 3)
    
    def _calculate_duration(self) -> str:
        """Calculate session duration"""
        start = datetime.fromisoformat(self.start_time)
        end = datetime.now()
        duration = end - start
        return str(duration) 