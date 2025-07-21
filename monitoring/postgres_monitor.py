import psycopg2
import psycopg2.extras
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import threading
import logging

class PostgresMonitor:
    def __init__(self, db_name="patent_monitoring", host="localhost", port=5432, user=None, password=None):
        self.db_name = db_name
        self.host = host
        self.port = port
        self.user = user
        self.password = password
        self.connection = None
        self.lock = threading.Lock()
        
        # Initialize database
        self._init_database()
        
    def _get_connection(self):
        """Get database connection"""
        if self.connection is None or self.connection.closed:
            self.connection = psycopg2.connect(
                dbname=self.db_name,
                host=self.host,
                port=self.port,
                user=self.user,
                password=self.password
            )
        return self.connection
    
    def _init_database(self):
        """Initialize database tables"""
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            # Create metrics table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS chat_metrics (
                    id SERIAL PRIMARY KEY,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    query_text TEXT,
                    response_text TEXT,
                    response_time_ms INTEGER,
                    tokens_used INTEGER,
                    model_name VARCHAR(100),
                    source_count INTEGER,
                    guardrail_scores JSONB,
                    evaluation_scores JSONB,
                    session_id VARCHAR(100),
                    user_id VARCHAR(100)
                )
            """)
            
            # Create system metrics table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS system_metrics (
                    id SERIAL PRIMARY KEY,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    metric_name VARCHAR(100),
                    metric_value REAL,
                    metric_unit VARCHAR(50),
                    additional_data JSONB
                )
            """)
            
            # Create performance metrics table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS performance_metrics (
                    id SERIAL PRIMARY KEY,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    component VARCHAR(100),
                    operation VARCHAR(100),
                    duration_ms INTEGER,
                    success BOOLEAN,
                    error_message TEXT,
                    additional_data JSONB
                )
            """)
            
            conn.commit()
            cursor.close()
            print("✅ PostgreSQL monitoring database initialized successfully")
            
        except Exception as e:
            print(f"❌ Error initializing PostgreSQL database: {e}")
            raise
    
    def record_chat_metric(self, query_text: str, response_text: str, response_time_ms: int,
                          tokens_used: int, model_name: str, source_count: int = 0,
                          guardrail_scores: Dict = None, evaluation_scores: Dict = None,
                          session_id: str = None, user_id: str = None):
        """Record a chat interaction metric"""
        try:
            with self.lock:
                conn = self._get_connection()
                cursor = conn.cursor()
                
                cursor.execute("""
                    INSERT INTO chat_metrics 
                    (query_text, response_text, response_time_ms, tokens_used, model_name, 
                     source_count, guardrail_scores, evaluation_scores, session_id, user_id)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """, (
                    query_text, response_text, response_time_ms, tokens_used, model_name,
                    source_count, json.dumps(guardrail_scores) if guardrail_scores else None,
                    json.dumps(evaluation_scores) if evaluation_scores else None,
                    session_id, user_id
                ))
                
                conn.commit()
                cursor.close()
                
        except Exception as e:
            print(f"❌ Error recording chat metric: {e}")
    
    def record_system_metric(self, metric_name: str, metric_value: float, 
                           metric_unit: str = None, additional_data: Dict = None):
        """Record a system metric"""
        try:
            with self.lock:
                conn = self._get_connection()
                cursor = conn.cursor()
                
                cursor.execute("""
                    INSERT INTO system_metrics (metric_name, metric_value, metric_unit, additional_data)
                    VALUES (%s, %s, %s, %s)
                """, (
                    metric_name, metric_value, metric_unit,
                    json.dumps(additional_data) if additional_data else None
                ))
                
                conn.commit()
                cursor.close()
                
        except Exception as e:
            print(f"❌ Error recording system metric: {e}")
    
    def record_performance_metric(self, component: str, operation: str, duration_ms: int,
                                success: bool, error_message: str = None, additional_data: Dict = None):
        """Record a performance metric"""
        try:
            with self.lock:
                conn = self._get_connection()
                cursor = conn.cursor()
                
                cursor.execute("""
                    INSERT INTO performance_metrics 
                    (component, operation, duration_ms, success, error_message, additional_data)
                    VALUES (%s, %s, %s, %s, %s, %s)
                """, (
                    component, operation, duration_ms, success, error_message,
                    json.dumps(additional_data) if additional_data else None
                ))
                
                conn.commit()
                cursor.close()
                
        except Exception as e:
            print(f"❌ Error recording performance metric: {e}")
    
    def get_chat_metrics(self, hours: int = 24) -> List[Dict]:
        """Get chat metrics from the last N hours"""
        try:
            conn = self._get_connection()
            cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
            
            cursor.execute("""
                SELECT * FROM chat_metrics 
                WHERE timestamp >= NOW() - INTERVAL '%s hours'
                ORDER BY timestamp DESC
            """, (hours,))
            
            results = cursor.fetchall()
            cursor.close()
            
            return [dict(row) for row in results]
            
        except Exception as e:
            print(f"❌ Error getting chat metrics: {e}")
            return []
    
    def get_system_metrics(self, hours: int = 24) -> List[Dict]:
        """Get system metrics from the last N hours"""
        try:
            conn = self._get_connection()
            cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
            
            cursor.execute("""
                SELECT * FROM system_metrics 
                WHERE timestamp >= NOW() - INTERVAL '%s hours'
                ORDER BY timestamp DESC
            """, (hours,))
            
            results = cursor.fetchall()
            cursor.close()
            
            return [dict(row) for row in results]
            
        except Exception as e:
            print(f"❌ Error getting system metrics: {e}")
            return []
    
    def get_performance_metrics(self, hours: int = 24) -> List[Dict]:
        """Get performance metrics from the last N hours"""
        try:
            conn = self._get_connection()
            cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
            
            cursor.execute("""
                SELECT * FROM performance_metrics 
                WHERE timestamp >= NOW() - INTERVAL '%s hours'
                ORDER BY timestamp DESC
            """, (hours,))
            
            results = cursor.fetchall()
            cursor.close()
            
            return [dict(row) for row in results]
            
        except Exception as e:
            print(f"❌ Error getting performance metrics: {e}")
            return []
    
    def get_metrics_summary(self, hours: int = 24) -> Dict:
        """Get a summary of all metrics"""
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            # Chat metrics summary
            cursor.execute("""
                SELECT 
                    COUNT(*) as total_queries,
                    AVG(response_time_ms) as avg_response_time,
                    AVG(tokens_used) as avg_tokens,
                    COUNT(DISTINCT session_id) as unique_sessions
                FROM chat_metrics 
                WHERE timestamp >= NOW() - INTERVAL '%s hours'
            """, (hours,))
            
            chat_summary = cursor.fetchone()
            
            # Performance metrics summary
            cursor.execute("""
                SELECT 
                    component,
                    AVG(duration_ms) as avg_duration,
                    COUNT(*) as total_operations,
                    SUM(CASE WHEN success THEN 1 ELSE 0 END) as successful_operations
                FROM performance_metrics 
                WHERE timestamp >= NOW() - INTERVAL '%s hours'
                GROUP BY component
            """, (hours,))
            
            performance_summary = cursor.fetchall()
            
            cursor.close()
            
            return {
                "chat_metrics": {
                    "total_queries": chat_summary[0] if chat_summary else 0,
                    "avg_response_time_ms": round(chat_summary[1], 2) if chat_summary and chat_summary[1] else 0,
                    "avg_tokens": round(chat_summary[2], 2) if chat_summary and chat_summary[2] else 0,
                    "unique_sessions": chat_summary[3] if chat_summary else 0
                },
                "performance_metrics": {
                    row[0]: {
                        "avg_duration_ms": round(row[1], 2) if row[1] else 0,
                        "total_operations": row[2],
                        "success_rate": round((row[3] / row[2]) * 100, 2) if row[2] > 0 else 0
                    }
                    for row in performance_summary
                }
            }
            
        except Exception as e:
            print(f"❌ Error getting metrics summary: {e}")
            return {}
    
    def cleanup_old_data(self, days: int = 30):
        """Clean up data older than N days"""
        try:
            with self.lock:
                conn = self._get_connection()
                cursor = conn.cursor()
                
                cursor.execute("""
                    DELETE FROM chat_metrics 
                    WHERE timestamp < NOW() - INTERVAL '%s days'
                """, (days,))
                
                cursor.execute("""
                    DELETE FROM system_metrics 
                    WHERE timestamp < NOW() - INTERVAL '%s days'
                """, (days,))
                
                cursor.execute("""
                    DELETE FROM performance_metrics 
                    WHERE timestamp < NOW() - INTERVAL '%s days'
                """, (days,))
                
                conn.commit()
                cursor.close()
                
                print(f"✅ Cleaned up data older than {days} days")
                
        except Exception as e:
            print(f"❌ Error cleaning up old data: {e}")
    
    def close(self):
        """Close database connection"""
        if self.connection:
            self.connection.close()
            print("✅ PostgreSQL connection closed")
    
    def get_real_time_metrics(self) -> Dict:
        """Get real-time metrics for dashboard display"""
        try:
            # Get recent chat metrics
            chat_metrics = self.get_chat_metrics(hours=1)
            
            # Get recent system metrics
            system_metrics = self.get_system_metrics(hours=1)
            
            # Get recent performance metrics
            performance_metrics = self.get_performance_metrics(hours=1)
            
            # Calculate real-time statistics
            total_queries = len(chat_metrics)
            avg_response_time = 0
            total_tokens = 0
            success_rate = 0
            
            if chat_metrics:
                response_times = [m['response_time_ms'] for m in chat_metrics if m['response_time_ms']]
                avg_response_time = sum(response_times) / len(response_times) if response_times else 0
                total_tokens = sum(m['tokens_used'] for m in chat_metrics if m['tokens_used'])
            
            if performance_metrics:
                successful_ops = len([m for m in performance_metrics if m['success']])
                success_rate = (successful_ops / len(performance_metrics)) * 100 if performance_metrics else 0
            
            return {
                'total_queries': total_queries,
                'avg_response_time_ms': avg_response_time,
                'total_tokens': total_tokens,
                'success_rate_percent': success_rate,
                'recent_metrics': {
                    'chat': chat_metrics[:10],  # Last 10 chat interactions
                    'system': system_metrics[:10],  # Last 10 system metrics
                    'performance': performance_metrics[:10]  # Last 10 performance metrics
                }
            }
            
        except Exception as e:
            print(f"❌ Error getting real-time metrics: {e}")
            return {
                'total_queries': 0,
                'avg_response_time_ms': 0,
                'total_tokens': 0,
                'success_rate_percent': 0,
                'recent_metrics': {'chat': [], 'system': [], 'performance': []}
            }
    
    def stop_background_monitoring(self):
        """Stop background monitoring (no-op for PostgresMonitor)"""
        # PostgresMonitor doesn't use background monitoring
        # This method exists for compatibility with other monitoring systems
        pass
    
    def start_monitoring(self, auto_open_dashboard=False):
        """Start monitoring (no-op for PostgresMonitor)"""
        # PostgresMonitor doesn't use background monitoring
        # This method exists for compatibility with other monitoring systems
        print("✅ PostgreSQL monitoring is active")
        if auto_open_dashboard:
            print("📊 Dashboard available via Grafana integration")
    
    def get_performance_summary(self, hours: int = 1) -> Dict:
        """Get performance summary for the last N hours"""
        try:
            conn = self._get_connection()
            cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
            
            cursor.execute("""
                SELECT 
                    component,
                    COUNT(*) as total_operations,
                    AVG(duration_ms) as avg_duration,
                    SUM(CASE WHEN success THEN 1 ELSE 0 END) as successful_operations,
                    MAX(timestamp) as last_operation
                FROM performance_metrics 
                WHERE timestamp >= NOW() - INTERVAL '%s hours'
                GROUP BY component
                ORDER BY avg_duration DESC
            """, (hours,))
            
            results = cursor.fetchall()
            cursor.close()
            
            summary = {}
            for row in results:
                summary[row['component']] = {
                    'total_operations': row['total_operations'],
                    'avg_duration_ms': round(row['avg_duration'], 2) if row['avg_duration'] else 0,
                    'success_rate': round((row['successful_operations'] / row['total_operations']) * 100, 2) if row['total_operations'] > 0 else 0,
                    'last_operation': row['last_operation'].isoformat() if row['last_operation'] else None
                }
            
            return summary
            
        except Exception as e:
            print(f"❌ Error getting performance summary: {e}")
            return {}
    
    def get_system_health_summary(self) -> Dict:
        """Get system health summary"""
        try:
            conn = self._get_connection()
            cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
            
            cursor.execute("""
                SELECT 
                    metric_name,
                    metric_value,
                    MAX(timestamp) as last_check
                FROM system_metrics 
                WHERE metric_name LIKE '%_status' 
                AND timestamp >= NOW() - INTERVAL '1 hour'
                GROUP BY metric_name, metric_value
            """)
            
            results = cursor.fetchall()
            cursor.close()
            
            health_summary = {}
            for row in results:
                service_name = row['metric_name'].replace('_status', '')
                health_summary[service_name] = {
                    'is_online': bool(row['metric_value']),
                    'last_check': row['last_check'].isoformat() if row['last_check'] else None
                }
            
            return health_summary
            
        except Exception as e:
            print(f"❌ Error getting system health summary: {e}")
            return {}
    
    def export_metrics_for_grafana(self) -> Dict:
        """Export metrics in Grafana-compatible format"""
        try:
            # Get recent metrics (last 24 hours)
            chat_metrics = self.get_chat_metrics(hours=24)
            performance_metrics = self.get_performance_metrics(hours=24)
            
            # Format for Grafana
            grafana_data = {
                'chat_metrics': {
                    'total_queries': len(chat_metrics),
                    'avg_response_time': sum(m['response_time_ms'] for m in chat_metrics if m['response_time_ms']) / len(chat_metrics) if chat_metrics else 0,
                    'total_tokens': sum(m['tokens_used'] for m in chat_metrics if m['tokens_used']),
                    'recent_interactions': chat_metrics[:50]  # Last 50 interactions
                },
                'performance_metrics': {
                    'total_operations': len(performance_metrics),
                    'success_rate': len([m for m in performance_metrics if m['success']]) / len(performance_metrics) * 100 if performance_metrics else 0,
                    'recent_operations': performance_metrics[:50]
                }
            }
            
            return grafana_data
            
        except Exception as e:
            print(f"❌ Error exporting metrics for Grafana: {e}")
            return {}
    
    def save_metrics_to_file(self, filename: str = None) -> str:
        """Save metrics to a JSON file"""
        try:
            import json
            from datetime import datetime
            
            if not filename:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"monitoring_data_{timestamp}.json"
            
            # Get all metrics
            chat_metrics = self.get_chat_metrics(hours=24)
            system_metrics = self.get_system_metrics(hours=24)
            performance_metrics = self.get_performance_metrics(hours=24)
            
            # Prepare data for export
            export_data = {
                'export_timestamp': datetime.now().isoformat(),
                'chat_metrics': chat_metrics,
                'system_metrics': system_metrics,
                'performance_metrics': performance_metrics,
                'summary': {
                    'total_chat_interactions': len(chat_metrics),
                    'total_system_metrics': len(system_metrics),
                    'total_performance_metrics': len(performance_metrics)
                }
            }
            
            # Save to file
            with open(filename, 'w') as f:
                json.dump(export_data, f, indent=2, default=str)
            
            print(f"✅ Monitoring data saved to: {filename}")
            return filename
            
        except Exception as e:
            print(f"❌ Error saving metrics to file: {e}")
            return ""

# Global monitor instance
postgres_monitor = PostgresMonitor() 