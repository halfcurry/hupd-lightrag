# Grafana Setup for Patent Chatbot Metrics

This guide explains how to set up Grafana to visualize session and evaluation metrics from the patent chatbot.

## 📊 Available Metrics Dashboards

### 1. PostgreSQL Monitoring Dashboard (Real-time)
- **File**: `monitoring/simple_postgres_dashboard.json`
- **Data Source**: PostgreSQL
- **Metrics**: Real-time chat metrics, response times, performance data
- **Refresh**: 30 seconds

### 2. Session & Evaluation Dashboard (Comprehensive)
- **File**: `monitoring/grafana_session_dashboard.json`
- **Data Source**: SQLite (sessions.db)
- **Metrics**: Session interactions, evaluation scores, data source usage
- **Refresh**: 30 seconds

## 🚀 Setup Instructions

### Step 1: Install Grafana

```bash
# macOS (using Homebrew)
brew install grafana

# Or download from https://grafana.com/grafana/download
```

### Step 2: Start Grafana

```bash
# Start Grafana service
brew services start grafana

# Or run directly
grafana-server --config=/usr/local/etc/grafana/grafana.ini
```

### Step 3: Access Grafana

1. Open browser: http://localhost:3000
2. Default credentials: admin/admin
3. Change password when prompted

### Step 4: Configure Data Sources

#### Option A: PostgreSQL Data Source (for real-time metrics)

1. Go to **Configuration** → **Data Sources**
2. Click **Add data source**
3. Select **PostgreSQL**
4. Configure:
   - **Name**: `patent-chatbot-postgres`
   - **Host**: `localhost:5432`
   - **Database**: `patent_chatbot_metrics`
   - **User**: `postgres`
   - **Password**: `your_password`
   - **SSL Mode**: `disable`

#### Option B: SQLite Data Source (for session metrics)

1. Go to **Configuration** → **Data Sources**
2. Click **Add data source**
3. Select **SQLite** (you may need to install the plugin)
4. Configure:
   - **Name**: `patent-chatbot-sqlite`
   - **Database Path**: `/path/to/your/patent_project/sessions.db`
   - **Cache Time**: `60s`

### Step 5: Import Dashboards

#### Import PostgreSQL Dashboard

1. Go to **Dashboards** → **Import**
2. Upload `monitoring/simple_postgres_dashboard.json`
3. Select PostgreSQL data source
4. Click **Import**

#### Import Session Dashboard

1. Go to **Dashboards** → **Import**
2. Upload `monitoring/grafana_session_dashboard.json`
3. Select SQLite data source
4. Click **Import**

## 📈 Dashboard Metrics

### PostgreSQL Dashboard (Real-time)
- **Response Time Over Time**: Real-time response times
- **Total Queries**: Count of queries in last 24 hours
- **Average Response Time**: Average response time
- **Performance Metrics**: System performance data

### Session Dashboard (Comprehensive)
- **Response Time Over Time**: Session interaction response times
- **Total Interactions**: Count of all interactions
- **Average Response Time**: Average session response time
- **Interactions per Hour**: Hourly interaction patterns
- **Evaluated Interactions**: Only LLM/RAG interactions with evaluation
- **Relevance Score Over Time**: Evaluation relevance scores
- **Coherence Score Over Time**: Evaluation coherence scores
- **Data Source Usage**: Breakdown by data source (lightrag, menu_selection, etc.)
- **Session Status**: Session completion status

## 🔧 SQLite Plugin Installation

If SQLite data source is not available:

```bash
# Install SQLite plugin
grafana-cli plugins install grafana-sqlite-datasource

# Restart Grafana
brew services restart grafana
```

## 📊 Key Metrics Explained

### Session Metrics
- **Total Interactions**: All chatbot interactions
- **Evaluated Interactions**: Only LLM/RAG interactions with evaluation scores
- **Data Source Breakdown**: 
  - `lightrag`: LLM/RAG interactions (evaluated)
  - `menu_selection`: Menu selections (not evaluated)
  - `patent_search`: Patent search (evaluated)
  - `patent_analysis`: Patent analysis (evaluated)

### Evaluation Metrics
- **Relevance Score**: How relevant the response is to the query (0-1)
- **Coherence Score**: How coherent and logical the response is (0-1)
- **Factual Accuracy**: Accuracy of factual information (0-1)
- **Completeness**: Completeness of the response (0-1)

### Performance Metrics
- **Response Time**: Time taken to generate responses
- **Session Duration**: How long sessions last
- **Interaction Frequency**: How often users interact

## 🎯 Dashboard Features

### Real-time Monitoring
- **Auto-refresh**: Dashboards refresh every 30 seconds
- **Time Range**: Default 24 hours, adjustable
- **Live Updates**: New interactions appear automatically

### Interactive Features
- **Time Range Selector**: Choose different time periods
- **Panel Zoom**: Click panels to expand
- **Drill-down**: Click data points for details
- **Export**: Export data as CSV/JSON

### Alerts (Optional)
Set up alerts for:
- High response times (>10 seconds)
- Low evaluation scores (<0.5)
- High error rates
- Session timeouts

## 🔍 Troubleshooting

### Common Issues

1. **Data Source Connection Failed**
   - Check database is running
   - Verify connection credentials
   - Check firewall settings

2. **No Data Showing**
   - Verify data source is configured correctly
   - Check time range settings
   - Ensure chatbot has generated data

3. **SQLite Plugin Not Available**
   - Install plugin manually
   - Restart Grafana
   - Check plugin compatibility

### Debug Commands

```bash
# Check if sessions.db has data
sqlite3 sessions.db "SELECT COUNT(*) FROM session_interactions;"

# Check recent interactions
sqlite3 sessions.db "SELECT * FROM session_interactions ORDER BY timestamp DESC LIMIT 5;"

# Check session metadata
sqlite3 sessions.db "SELECT * FROM session_metadata ORDER BY start_time DESC LIMIT 5;"
```

## 📱 Mobile Access

- Grafana is mobile-responsive
- Access via: http://your-server-ip:3000
- Use Grafana mobile app for notifications

## 🔐 Security

- Change default admin password
- Use HTTPS in production
- Set up user roles and permissions
- Regular backups of dashboards and data sources

## 📈 Advanced Features

### Custom Queries
Create custom SQL queries for specific metrics:

```sql
-- Average evaluation scores by data source
SELECT data_source, 
       AVG(json_extract(evaluation_scores, '$.relevance_score')) as avg_relevance,
       AVG(json_extract(evaluation_scores, '$.coherence_score')) as avg_coherence
FROM session_interactions 
WHERE evaluation_scores IS NOT NULL 
GROUP BY data_source;
```

### Annotations
Add annotations for important events:
- Chatbot updates
- System maintenance
- Performance issues

### Variables
Create dashboard variables for:
- Time ranges
- Data sources
- Session IDs
- Evaluation thresholds

## 🎉 Success Indicators

You'll know it's working when you see:
- ✅ Real-time data flowing into dashboards
- ✅ Response time graphs showing patterns
- ✅ Evaluation scores for LLM/RAG interactions
- ✅ Session completion status updates
- ✅ Data source usage breakdowns

## 📞 Support

For issues with:
- **Grafana Setup**: Check Grafana documentation
- **Data Sources**: Verify database connections
- **Dashboard Import**: Check JSON format and data source UIDs
- **Metrics**: Ensure chatbot is generating data 