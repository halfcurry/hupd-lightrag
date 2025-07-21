-- SQLite to PostgreSQL Migration Script
-- This script creates PostgreSQL tables that mirror the SQLite session database structure

-- Create session_metadata table
CREATE TABLE IF NOT EXISTS session_metadata (
    session_id VARCHAR(100) PRIMARY KEY,
    start_time TIMESTAMP,
    end_time TIMESTAMP,
    total_interactions INTEGER DEFAULT 0,
    evaluation_status VARCHAR(50) DEFAULT 'pending',
    overall_score FLOAT DEFAULT 0.0,
    detailed_metrics JSONB
);

-- Create session_interactions table
CREATE TABLE IF NOT EXISTS session_interactions (
    id SERIAL PRIMARY KEY,
    session_id VARCHAR(100),
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    query_text TEXT,
    response_text TEXT,
    data_source VARCHAR(50),
    response_time_ms INTEGER,
    guardrail_scores JSONB,
    evaluation_scores JSONB,
    interaction_type VARCHAR(50) DEFAULT 'llm_rag',
    session_phase VARCHAR(50) DEFAULT 'active'
);

-- Create indexes for better performance
CREATE INDEX IF NOT EXISTS idx_session_interactions_session_id ON session_interactions(session_id);
CREATE INDEX IF NOT EXISTS idx_session_interactions_timestamp ON session_interactions(timestamp);
CREATE INDEX IF NOT EXISTS idx_session_interactions_data_source ON session_interactions(data_source);

-- Create a function to extract JSON values for evaluation scores
CREATE OR REPLACE FUNCTION extract_evaluation_score(evaluation_scores JSONB, score_key TEXT)
RETURNS FLOAT AS $$
BEGIN
    RETURN (evaluation_scores->>score_key)::FLOAT;
EXCEPTION
    WHEN OTHERS THEN
        RETURN 0.0;
END;
$$ LANGUAGE plpgsql;

-- Create views for easier querying
CREATE OR REPLACE VIEW evaluation_metrics AS
SELECT 
    session_id,
    timestamp,
    data_source,
    extract_evaluation_score(evaluation_scores, 'relevance_score') as relevance_score,
    extract_evaluation_score(evaluation_scores, 'coherence_score') as coherence_score,
    extract_evaluation_score(evaluation_scores, 'factual_accuracy') as factual_accuracy,
    extract_evaluation_score(evaluation_scores, 'completeness') as completeness,
    response_time_ms
FROM session_interactions 
WHERE evaluation_scores IS NOT NULL;

-- Create view for data source statistics
CREATE OR REPLACE VIEW data_source_stats AS
SELECT 
    data_source,
    COUNT(*) as total_interactions,
    COUNT(CASE WHEN evaluation_scores IS NOT NULL THEN 1 END) as evaluated_interactions,
    AVG(response_time_ms) as avg_response_time,
    AVG(extract_evaluation_score(evaluation_scores, 'relevance_score')) as avg_relevance_score,
    AVG(extract_evaluation_score(evaluation_scores, 'coherence_score')) as avg_coherence_score
FROM session_interactions 
GROUP BY data_source;

-- Create view for hourly statistics
CREATE OR REPLACE VIEW hourly_stats AS
SELECT 
    DATE_TRUNC('hour', timestamp) as hour,
    COUNT(*) as total_interactions,
    COUNT(CASE WHEN evaluation_scores IS NOT NULL THEN 1 END) as evaluated_interactions,
    AVG(response_time_ms) as avg_response_time
FROM session_interactions 
GROUP BY DATE_TRUNC('hour', timestamp)
ORDER BY hour; 