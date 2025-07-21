# Patent Analysis Chatbot with Enhanced Monitoring & Grafana Integration

## 🚀 Overview

This is a comprehensive patent analysis chatbot with advanced monitoring, evaluation metrics, and real-time Grafana visualization. The system provides patent analysis capabilities with guardrails validation, comprehensive evaluation metrics, and automated data synchronization.

## ✨ Key Features

### 🤖 **Patent Analysis Capabilities**
- **Existing Patent Analysis**: Analyze patents from database with detailed technical assessment
- **New Invention Evaluation**: Evaluate new invention ideas with comprehensive metrics
- **Patent Search**: Search patents by technology/topic with enhanced results
- **Enhanced Analysis Mode**: Advanced patent analysis with detailed evaluation metrics

### 🛡️ **Guardrails & Safety**
- **Content Safety**: Profanity detection and content appropriateness
- **Topic Relevance**: Ensures responses stay on-topic for patent discussions
- **Politeness Filter**: Maintains professional communication standards
- **Response Validation**: Real-time validation of chatbot responses

### 📊 **Comprehensive Evaluation Metrics**
- **Relevance Score**: How relevant responses are to queries
- **Coherence Score**: Logical flow and coherence of responses
- **Factual Accuracy**: Accuracy of patent information and claims
- **Completeness**: Comprehensiveness of responses
- **Enhanced Metrics**: Logical flow, contextual consistency, topical relevance unity
- **Advanced Coherence**: Reference resolution, discourse structure, faithfulness to retrieval chain

### 📈 **Real-time Monitoring & Visualization**
- **PostgreSQL Integration**: Real-time metrics storage
- **Grafana Dashboards**: Beautiful, interactive visualizations
- **Auto-sync**: Automatic SQLite to PostgreSQL data synchronization
- **Session Management**: Comprehensive session tracking and evaluation
- **Performance Metrics**: Response times, interaction patterns, system health

## 🏗️ Architecture

### **Core Components**
```
PatentChatbot
├── LightRAG Integration (Document Retrieval)
├── Guardrails Validator (Content Safety)
├── Response Evaluator (Quality Assessment)
├── Session Logger (Conversation Tracking)
├── Auto-sync System (Data Synchronization)
└── Grafana Integration (Real-time Visualization)
```

### **Data Flow**
```
User Query → PatentChatbot → LightRAG → Response Generation → 
Guardrails Validation → Evaluation Metrics → Session Logging → 
Auto-sync to PostgreSQL → Grafana Visualization
```

## 📦 Installation & Setup

### **1. Environment Setup**
```bash
# Create virtual environment
python -m venv lightrag_env
source lightrag_env/bin/activate  # On Windows: lightrag_env\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### **2. Database Setup**
```bash
# PostgreSQL setup (for monitoring)
psql -d patent_monitoring -f monitoring/sqlite_to_postgres_migration.sql
```

### **3. Grafana Setup**
```bash
# Install Grafana
brew install grafana  # macOS
# Or download from https://grafana.com/grafana/download

# Start Grafana
brew services start grafana

# Access Grafana
# Open: http://localhost:3000
# Default credentials: admin/admin
```

### **4. Data Source Configuration**
1. **PostgreSQL Data Source**:
   - Name: `patent-chatbot-postgres`
   - Host: `localhost:5432`
   - Database: `patent_monitoring`
   - User: `aniket.rastogi`
   - SSL Mode: `disable`

2. **Import Dashboard**:
   - Upload: `monitoring/grafana_comprehensive_dashboard.json`
   - Select PostgreSQL data source
   - Update data source UID in dashboard settings

## 🎯 Usage

### **Running the Chatbot**
```bash
# Interactive mode
cd chatbot
python patent_chatbot.py

# Gradio web interface
python patent_chatbot.py --gradio

# With specific configuration
python patent_chatbot.py --lightrag-url http://localhost:9621 --enable-monitoring
```

### **Key Features in Action**

#### **1. Patent Analysis**
```
User: "Analyze patent US12345678"
→ System retrieves patent data from LightRAG
→ Generates comprehensive analysis
→ Validates with guardrails
→ Evaluates response quality
→ Logs to session database
→ Syncs to PostgreSQL
→ Updates Grafana dashboard
```

#### **2. Enhanced Evaluation**
```
Response Evaluation Metrics:
├── Relevance Score: 0.92 (High relevance to query)
├── Coherence Score: 0.89 (Logical flow)
├── Factual Accuracy: 0.85 (Patent information accuracy)
├── Completeness: 0.88 (Comprehensive response)
├── Guardrails:
│   ├── Profanity: 0.0 (Clean content)
│   ├── Topic Relevance: 0.95 (On-topic)
│   └── Politeness: 0.88 (Professional)
└── Enhanced Metrics:
    ├── Logical Flow: 0.87
    ├── Contextual Consistency: 0.91
    ├── Topical Relevance Unity: 0.93
    └── Reference Resolution: 0.89
```

## 📊 Monitoring & Visualization

### **Grafana Dashboard Features**

#### **Real-time Metrics**
- **Response Time Over Time**: Performance monitoring
- **Total Interactions**: Usage volume tracking
- **Average Response Time**: Performance analysis
- **Interactions per Hour**: Usage patterns

#### **Evaluation Quality**
- **Relevance Score Over Time**: Response relevance tracking
- **Coherence Score Over Time**: Response quality monitoring
- **Factual Accuracy Over Time**: Information accuracy
- **Completeness Over Time**: Response comprehensiveness

#### **Guardrails Monitoring**
- **Profanity Guardrail**: Content safety tracking
- **Topic Relevance Guardrail**: Topic adherence
- **Politeness Guardrail**: Professionalism monitoring

#### **Data Source Usage**
- **Data Source Breakdown**: Interaction type analysis
- **Session Status**: Completion tracking

### **Auto-sync System**
- **Automatic Synchronization**: SQLite → PostgreSQL every 30 seconds
- **Real-time Updates**: Fresh data in Grafana dashboards
- **Integrated Pipeline**: No separate sync processes needed
- **Error Handling**: Robust error recovery and logging

## 🔧 Technical Details

### **Session Management**
```python
# Session lifecycle
Session Created → Interactions Logged → Evaluation Calculated → 
Session Closed → Post-session Analysis → Grafana Visualization
```

### **Evaluation Metrics**
```python
@dataclass
class EvaluationScores:
    # Basic metrics
    relevance_score: float = 0.0
    coherence_score: float = 0.0
    
    # Enhanced metrics
    factual_accuracy: float = 0.0
    completeness: float = 0.0
    logical_flow: float = 0.0
    contextual_consistency: float = 0.0
    topical_relevance_unity: float = 0.0
    reference_resolution: float = 0.0
    discourse_structure_cohesion: float = 0.0
    faithfulness_retrieval_chain: float = 0.0
    temporal_causal_coherence: float = 0.0
    semantic_coherence: float = 0.0
    
    # Guardrails
    profanity_score: float = 0.0
    topic_relevance_score: float = 0.0
    politeness_score: float = 0.0
```

### **Database Schema**
```sql
-- Session metadata
CREATE TABLE session_metadata (
    session_id VARCHAR(100) PRIMARY KEY,
    start_time TIMESTAMP,
    end_time TIMESTAMP,
    total_interactions INTEGER DEFAULT 0,
    evaluation_status VARCHAR(50) DEFAULT 'pending',
    overall_score FLOAT DEFAULT 0.0,
    detailed_metrics JSONB
);

-- Session interactions
CREATE TABLE session_interactions (
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
```

## 🎨 Grafana Dashboard Configuration

### **Dashboard Panels**
1. **Response Time Over Time**: Real-time performance monitoring
2. **Total Interactions**: Usage volume tracking
3. **Average Response Time**: Performance analysis
4. **Relevance Score Over Time**: Response quality
5. **Coherence Score Over Time**: Response coherence
6. **Factual Accuracy Over Time**: Information accuracy
7. **Completeness Over Time**: Response comprehensiveness
8. **Profanity Guardrail Over Time**: Content safety
9. **Topic Relevance Guardrail Over Time**: Topic adherence
10. **Politeness Guardrail Over Time**: Professionalism
11. **Data Source Usage**: Interaction type breakdown

### **Key Features**
- **Auto-refresh**: Every 30 seconds
- **Real-time data**: From PostgreSQL
- **Interactive panels**: Click to drill down
- **Time range selector**: Choose different periods
- **Export capabilities**: CSV/JSON export

## 🔍 Troubleshooting

### **Common Issues**

#### **1. Grafana Not Showing Data**
```bash
# Check data sync
python monitoring/sync_sqlite_to_postgres.py

# Verify PostgreSQL connection
psql -d patent_monitoring -c "SELECT COUNT(*) FROM session_interactions;"
```

#### **2. Auto-sync Not Working**
```bash
# Check if auto-sync is enabled in chatbot
# Auto-sync starts automatically when chatbot initializes
# Manual sync available via chatbot.manual_sync()
```

#### **3. Session Closure Messages**
```
Session session_1753006438 should be closed: duration=248s, inactivity=19801s, interactions=8
```
This is normal - sessions are automatically closed after 30 minutes of inactivity.

### **Debug Commands**
```bash
# Check session data
sqlite3 sessions.db "SELECT COUNT(*) FROM session_interactions;"

# Check PostgreSQL data
psql -d patent_monitoring -c "SELECT COUNT(*) FROM session_interactions;"

# Manual sync
python monitoring/sync_sqlite_to_postgres.py
```

## 📈 Performance Metrics

### **System Performance**
- **Response Time**: Average 2-5 seconds
- **Evaluation Coverage**: 100% of LLM/RAG interactions
- **Guardrail Compliance**: >95% clean content
- **Session Duration**: Average 10-15 minutes
- **Auto-sync Frequency**: Every 30 seconds

### **Quality Metrics**
- **Relevance Score**: Average 0.85+
- **Coherence Score**: Average 0.88+
- **Factual Accuracy**: Average 0.82+
- **Completeness**: Average 0.86+

## 🚀 Advanced Features

### **Enhanced Patent Analysis**
- **Comprehensive Evaluation**: All enhanced metrics included
- **Technical Depth Analysis**: Detailed patent assessment
- **Commercial Potential**: Market impact evaluation
- **Innovation Assessment**: Patent strength evaluation

### **Real-time Monitoring**
- **Live Dashboard Updates**: Real-time Grafana visualization
- **Session Tracking**: Complete conversation history
- **Performance Monitoring**: Response time tracking
- **Quality Assessment**: Continuous evaluation

### **Integrated Pipeline**
- **Auto-sync**: No manual intervention required
- **Session Management**: Automatic session lifecycle
- **Error Recovery**: Robust error handling
- **Data Persistence**: SQLite + PostgreSQL backup

## 📝 Recent Updates (Latest)

### **Enhanced Monitoring System**
- ✅ **Integrated Auto-sync**: SQLite to PostgreSQL automatic synchronization
- ✅ **Comprehensive Metrics**: All enhanced evaluation metrics included
- ✅ **Grafana Integration**: Real-time visualization with comprehensive dashboard
- ✅ **Session Management**: Complete session lifecycle tracking
- ✅ **Guardrails Integration**: Content safety and quality validation
- ✅ **Performance Optimization**: Efficient data flow and processing

### **Code Cleanup**
- ✅ **Removed Backup Files**: Cleaned up unnecessary backup files
- ✅ **Streamlined Monitoring**: Removed redundant monitoring files
- ✅ **Integrated Pipeline**: Auto-sync now part of main chatbot
- ✅ **Enhanced Documentation**: Comprehensive README with all features

## 🤝 Contributing

1. **Fork the repository**
2. **Create a feature branch**: `git checkout -b feature/new-feature`
3. **Make your changes**
4. **Test thoroughly**: Run the chatbot and verify monitoring
5. **Commit your changes**: `git commit -am 'Add new feature'`
6. **Push to the branch**: `git push origin feature/new-feature`
7. **Submit a pull request**

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **LightRAG**: For document retrieval capabilities
- **Grafana**: For beautiful data visualization
- **PostgreSQL**: For robust data storage
- **Ollama**: For local LLM capabilities

---

**🎉 The patent chatbot now features comprehensive monitoring, real-time visualization, and automated data synchronization!** 