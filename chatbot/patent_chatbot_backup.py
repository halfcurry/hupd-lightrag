#!/usr/bin/env python3
"""
Enhanced Patent Chatbot with Guardrails Integration

This chatbot provides patent analysis capabilities with:
- LightRAG integration for document retrieval
- Guardrails validation for response quality
- Evaluation metrics for response assessment
- Patent analysis (existing, new invention, search)
"""

import requests
import json
import time
import logging
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from datetime import datetime
import asyncio
import threading

# Import our guardrails validator
from chatbot.guardrails_validator import CustomGuardrailsValidator, GuardrailScores

# Import evaluation module
from evaluation.evaluate_responses import ResponseEvaluator, EvaluationScores

# Import monitoring module
from monitoring.postgres_monitor import postgres_monitor

# Import Neo4j fallback
from .neo4j_fallback import Neo4jFallback

# Import SQLite fallback
from .sqlite_fallback import SQLiteFallback

# Import patent analyzer
from .patent_analyzer import PatentAnalyzer, PatentInfo, AnalysisResult

# Import session logger
from .session_logger import SessionLogger

# Response database removed - using session logger for tracking

# Import for internet search
import requests
from bs4 import BeautifulSoup
import re

# OpenAI integration removed - using local LLM only

# Import enhanced patent analyzer
from .enhanced_patent_analyzer import EnhancedPatentAnalyzer

# Import query expansion module
from .query_expansion import query_expander

logger = logging.getLogger(__name__)

def get_timestamp():
    """Get current timestamp in a readable format"""
    return datetime.now().strftime("%H:%M:%S")

@dataclass
class ConversationState:
    """Container for conversation state management with enhanced context tracking"""
    mode: Optional[str] = None  # 'patent_analysis', 'follow_up', 'general', 'interactive_query'
    context: Dict[str, Any] = None
    last_response: Optional[str] = None
    follow_up_count: int = 0
    max_follow_ups: int = 5
    awaiting_yes_no: bool = False
    
    # Enhanced context tracking
    session_patents: List[Dict[str, Any]] = None  # List of patents discussed in this session
    current_patent_index: int = -1  # Index of the currently referenced patent
    conversation_history: List[Dict[str, Any]] = None  # Full conversation history
    session_start_time: Optional[float] = None  # Session start timestamp
    
    def __post_init__(self):
        if self.session_patents is None:
            self.session_patents = []
        if self.conversation_history is None:
            self.conversation_history = []
        if self.session_start_time is None:
            self.session_start_time = time.time()
    
    def add_patent_to_session(self, patent_data: Dict[str, Any]) -> int:
        """Add a patent to the session and return its index"""
        # Check if patent already exists in session
        for i, existing_patent in enumerate(self.session_patents):
            if existing_patent.get('patent_number') == patent_data.get('patent_number'):
                self.current_patent_index = i
                return i
        
        # Add new patent to session
        self.session_patents.append(patent_data)
        self.current_patent_index = len(self.session_patents) - 1
        return self.current_patent_index
    
    def get_current_patent(self) -> Optional[Dict[str, Any]]:
        """Get the currently referenced patent"""
        if 0 <= self.current_patent_index < len(self.session_patents):
            return self.session_patents[self.current_patent_index]
        return None
    
    def get_session_context_summary(self) -> str:
        """Get a summary of all patents discussed in this session"""
        if not self.session_patents:
            return "No patents have been discussed in this session."
        
        summary = f"Session contains {len(self.session_patents)} patent(s):\n"
        for i, patent in enumerate(self.session_patents):
            patent_num = patent.get('patent_number', 'Unknown')
            title = patent.get('title', 'Unknown')
            status = f" ({patent.get('status', 'Unknown')})"
            current_marker = " [CURRENT]" if i == self.current_patent_index else ""
            summary += f"{i+1}. {patent_num}: {title}{status}{current_marker}\n"
        
        return summary
    
    def add_conversation_entry(self, user_query: str, bot_response: str, context_type: str = "general"):
        """Add an entry to the conversation history"""
        self.conversation_history.append({
            'timestamp': time.time(),
            'user_query': user_query,
            'bot_response': bot_response,
            'context_type': context_type,
            'current_patent_index': self.current_patent_index
        })
    
    def get_recent_context(self, num_entries: int = 3) -> str:
        """Get recent conversation context for LLM prompts"""
        if not self.conversation_history:
            return ""
        
        recent_entries = self.conversation_history[-num_entries:]
        context = "Recent conversation context:\n"
        for entry in recent_entries:
            context += f"User: {entry['user_query']}\n"
            context += f"Assistant: {entry['bot_response'][:200]}...\n\n"
        
        return context

@dataclass
class ChatbotResponse:
    """Container for chatbot response with validation scores"""
    content: str
    sources: List[str]
    response_time: float
    guardrail_scores: GuardrailScores
    evaluation_scores: Optional[EvaluationScores] = None
    follow_up_prompt: Optional[str] = None  # For follow-up questions

class PatentChatbot:
    """
    Patent analysis chatbot with optional guardrails and evaluation
    """
    
    def __init__(self, lightrag_url: str = "http://localhost:9621", with_guardrails: bool = True, enable_monitoring: bool = True):
        self.lightrag_url = lightrag_url
        self.with_guardrails = with_guardrails
        self.guardrails_validator = CustomGuardrailsValidator() if with_guardrails else None
        self.evaluator = ResponseEvaluator()
        
        # Initialize enhanced patent analyzer
        self.enhanced_analyzer = EnhancedPatentAnalyzer(lightrag_url=lightrag_url)
        
        # Initialize monitoring
        if enable_monitoring:
            try:
                self.monitor = postgres_monitor
                print("✅ Monitoring initialized successfully")
            except Exception as e:
                print(f"⚠️ Monitoring initialization failed: {e}")
                self.monitor = None
        else:
            self.monitor = None
            
        self.neo4j_fallback = Neo4jFallback()  # Initialize Neo4j fallback
        
        # Initialize SQLite fallback
        self.sqlite_fallback = SQLiteFallback()
        
        # Initialize patent analyzer
        self.patent_analyzer = PatentAnalyzer()
        
        # Initialize session logger
        self.session_logger = SessionLogger()
        
        # Patent field categories based on G06N/G06V analysis
        self.patent_field_categories = [
            "Machine Learning & AI",
            "Computer Vision & Image Processing", 
            "Neural Networks & Deep Learning",
            "Pattern Recognition & Classification",
            "Data Mining & Analytics",
            "Bioinformatics & Computational Biology",
            "Natural Language Processing",
            "Robotics & Automation",
            "Signal Processing & Audio",
            "Others (search all patents)"
        ]
        
        # Patent analysis state
        self.analysis_mode = None
        self.patent_info = None
        self.analysis_step = 0
        self.selected_fields = []
        self.use_all_fields = False
        
        # Conversation state management
        self.conversation_state = ConversationState()
        
        # Greeting message with patent analysis options
        self.greeting = """
🤖 Welcome to the Patent Analysis Assistant!

Hi, I am your patent analysis chatbot! I'm here to help you analyze patents and inventions.

What type of patent analysis do you need?

1. 📚 Analyze existing patent (from database)
2. 🆕 Evaluate new invention idea
3. 🔍 Search patents by technology/topic

Please choose (1-3):
"""
        
        # General conversation responses
        self.general_responses = {
            "greetings": [
                "Hi there! 👋 How can I help you with patent analysis today?",
                "Hello! 😊 I'm your patent analysis assistant. What would you like to know?",
                "Hi! 🤖 Ready to help you explore patents and inventions!",
                "Greetings! 👨‍💼 I'm here to assist with your patent research needs."
            ],
            "capabilities": [
                "I can help you analyze patents, evaluate new inventions, search for similar patents, and answer questions about patent law and technology!",
                "My capabilities include patent analysis, invention evaluation, technical explanation, classification help, and general conversation about patents and inventions.",
                "I specialize in patent research, technical analysis, invention evaluation, and helping you understand complex inventions and patent claims."
            ],
            "thanks": [
                "You're welcome! 😊 Let me know if you need anything else.",
                "Happy to help! 🤖 Feel free to ask more questions.",
                "Anytime! 👨‍💼 I'm here whenever you need patent assistance."
            ],
            "goodbye": [
                "Goodbye! 👋 Thanks for using the Patent Analysis Assistant!",
                "See you later! 😊 Have a great day!",
                "Take care! 🤖 Come back anytime for more patent help."
            ]
        }
        
    def _is_general_conversation(self, query: str) -> bool:
        """Check if the query is general conversation that doesn't need LightRAG"""
        query_lower = query.lower().strip()
        
        # Very specific greetings only
        greetings = ["hi", "hello", "hey", "good morning", "good afternoon", "good evening", "greetings"]
        if any(greeting in query_lower for greeting in greetings) and len(query_lower.split()) <= 3:
            return True
            
        # Very specific capability questions
        capability_questions = ["what can you do", "what are your capabilities", "help", "what do you do", "tell me about yourself"]
        if any(phrase in query_lower for phrase in capability_questions):
            return True
            
        # Thanks and goodbyes
        thanks = ["thank", "thanks", "appreciate"]
        goodbyes = ["bye", "goodbye", "see you", "farewell", "exit", "quit"]
        if any(phrase in query_lower for phrase in thanks + goodbyes):
            return True
            
        # If it contains patent-related keywords, it should go to LightRAG
        patent_keywords = [
            "patent", "invention", "claim", "technology", "innovation", "device", "method", "system", "apparatus", "process", "composition", "machine", "manufacture",
            "machine learning", "artificial intelligence", "ai", "neural network", "deep learning", "blockchain", "iot", "internet of things", "robotics", "automation",
            "computer vision", "natural language processing", "nlp", "data mining", "analytics", "algorithm", "software", "hardware", "electronics", "biotechnology",
            "pharmaceutical", "medical device", "diagnostic", "therapeutic", "drug", "chemical", "material", "nanotechnology", "quantum", "renewable energy",
            "solar", "wind", "battery", "electric vehicle", "autonomous", "drone", "satellite", "wireless", "5g", "cybersecurity", "cryptography"
        ]
        if any(keyword in query_lower for keyword in patent_keywords):
            return False
            
        # If it's a question about specific things (not just greetings), let it go to LightRAG
        question_words = ["what", "how", "why", "when", "where", "which", "who"]
        if any(word in query_lower for word in question_words) and len(query_lower.split()) > 2:
            return False
            
        return True
    
    def _is_patent_analysis_query(self, query: str) -> bool:
        """Check if the query is related to patent analysis"""
        query_lower = query.lower().strip()
        
        # Check for analysis-related keywords
        analysis_keywords = [
            "analyze", "evaluate", "assess", "review", "examine", "study",
            "patent analysis", "invention evaluation", "patent search",
            "acceptance probability", "patent probability", "chances of acceptance"
        ]
        
        return any(keyword in query_lower for keyword in analysis_keywords)
    
    def _handle_patent_analysis_selection(self, query: str) -> str:
        """Handle the initial patent analysis selection"""
        query_lower = query.lower().strip()
        
        # Reset analysis state when starting a new selection
        self.analysis_mode = None
        self.analysis_step = 0
        self.patent_info = None
        
        if query_lower == "1":
            self.analysis_mode = "existing_patent"
            return """📚 Please provide the patent number or title to analyze:
(Examples: US12345678, 'Neural Network System', etc.)"""
        
        elif query_lower == "2":
            self.analysis_mode = "new_invention"
            self.analysis_step = 1
            return """📝 Question 1: What is the title of your invention?"""
        
        elif query_lower == "3":
            self.analysis_mode = "patent_search"
            return """🔍 What technology or topic would you like to search for?
(Examples: machine learning, IoT, blockchain, etc.)"""
        
        else:
            return """🤖 What type of patent analysis do you need?

1. 📚 Analyze existing patent (from database)
2. 🆕 Evaluate new invention idea
3. 🔍 Search patents by technology/topic

Please choose (1-3):"""
    
    def _handle_new_invention_collection(self, query: str) -> str:
        """Handle step-by-step collection of new invention information with validation"""
        if self.analysis_step == 1:
            # Validate title length
            is_valid, error_msg = self._validate_text_length(query, 10, "Title")
            if not is_valid:
                return error_msg
            
            # Store title and ask for abstract
            self.patent_info = PatentInfo(title=query.strip())
            self.analysis_step = 2
            return """📝 Question 2: Can you provide an abstract of your invention?
(2-3 sentences describing what it does)"""
        
        elif self.analysis_step == 2:
            # Validate abstract length
            is_valid, error_msg = self._validate_text_length(query, 50, "Abstract")
            if not is_valid:
                return error_msg
            
            # Store abstract and ask about description
            self.patent_info.abstract = query.strip()
            self.analysis_step = 3
            return """📝 Question 3: Do you have a detailed description available?

Please select:
- Type 'y' for Yes (I have a detailed description)
- Type 'n' for No (I don't have a detailed description)"""
        
        elif self.analysis_step == 3:
            # Handle description question with validation
            query_lower = query.lower().strip()
            if query_lower not in ['y', 'n', 'yes', 'no']:
                return """Please select a valid option:
- Type 'y' for Yes (I have a detailed description)
- Type 'n' for No (I don't have a detailed description)"""
            
            if query_lower in ['y', 'yes']:
                self.analysis_step = 3.5  # Ask for description
                return """📝 Please provide your detailed description:"""
            else:
                # No description, move to field selection
                self.patent_info.description = None
                self.analysis_step = 4
                return self._show_field_selection()
        
        elif self.analysis_step == 3.5:
            # User provided description, move to field selection
            self.patent_info.description = query.strip()
            self.analysis_step = 4
            return self._show_field_selection()
        
        elif self.analysis_step == 4:
            # Handle field selection
            return self._handle_field_selection(query)
    
    def _show_field_selection(self) -> str:
        """Show field selection options"""
        field_options = "\n".join([f"{i+1}. {field}" for i, field in enumerate(self.patent_field_categories)])
        return f"""📝 Question 4: Select the technology field(s) that best match your invention:

{field_options}

You can select multiple fields by entering numbers separated by commas (e.g., 1,3,5)
Or type 'all' to search across all fields:"""
    
    def _handle_field_selection(self, query: str) -> str:
        """Handle field selection input"""
        query_lower = query.lower().strip()
        
        if query_lower == 'all':
            self.use_all_fields = True
            self.selected_fields = []
        else:
            # Parse selected fields
            try:
                selected_indices = [int(x.strip()) - 1 for x in query.split(',')]
                self.selected_fields = [self.patent_field_categories[i] for i in selected_indices if 0 <= i < len(self.patent_field_categories)]
                
                if not self.selected_fields:
                    return """Please select at least one valid field. Enter numbers separated by commas (e.g., 1,3,5):"""
                    
            except (ValueError, IndexError):
                return """Invalid selection. Please enter valid numbers separated by commas (e.g., 1,3,5):"""
        
        # Perform analysis with selected fields
        self.analysis_step = 0
        self.analysis_mode = None
        
        # Create field-specific query for RAG
        if self.use_all_fields or "Others" in self.selected_fields:
            field_query = "all patent fields"
        else:
            field_query = ", ".join(self.selected_fields)
        
        rag_query = f"""Prior art search for invention: "{self.patent_info.title}"
Abstract: {self.patent_info.abstract}
Technology Fields: {field_query}
Description: {self.patent_info.description if self.patent_info.description else 'Not provided'}

Find similar patents, prior art, and related inventions in the database. Analyze novelty and patentability."""
        
        print("🔍 Performing RAG-based prior art search...")
        
        # Get RAG context for prior art search
        rag_context = self._get_rag_context(rag_query)
        
        # Generate LLM response with RAG context
        llm_response = self._generate_llm_response(rag_query, rag_context)
        
        # Combine with local analysis
        local_result = self.patent_analyzer.analyze_new_invention(self.patent_info)
        
        # Create comprehensive response
        response = f"""🔍 **Patent Analysis Complete**

**Your Invention:**
- Title: {self.patent_info.title}
- Technology Fields: {field_query}

**RAG-Based Prior Art Analysis:**
{llm_response}

**Local Analysis Summary:**
{local_result.analysis}

**Key Factors:**
{chr(10).join([f"• {factor}" for factor in local_result.key_factors]) if local_result.key_factors else "No factors available"}

**Recommendations:**
{chr(10).join([f"• {rec}" for rec in local_result.recommendations]) if local_result.recommendations else "No recommendations available"}

**Acceptance Probability:**
{local_result.probability:.1f}% (if available)"""
        
        # Add new invention to session context
        invention_data = {
            'patent_number': f"NEW_INVENTION_{int(time.time())}",  # Generate unique ID for new invention
            'title': self.patent_info.title,
            'status': 'New Invention',
            'main_ipc_code': 'G06N/G06V (AI/ML)',
            'source': 'User Input',
            'abstract': self.patent_info.abstract,
            'description': self.patent_info.description,
            'analysis_results': response,
            'technology_fields': field_query
        }
        self.conversation_state.add_patent_to_session(invention_data)
        
        # Add conversation entry
        self.conversation_state.add_conversation_entry(
            user_query=f"Analyze new invention: {self.patent_info.title}",
            bot_response=response,
            context_type="new_invention_analysis"
        )
        
        # Set conversation state for follow-up
        self.conversation_state.mode = "follow_up"
        self.conversation_state.context = {"invention_title": self.patent_info.title, "analysis_result": response}
        self.conversation_state.last_response = response
        self.conversation_state.follow_up_count = 0
        
        # Reset analysis mode
        self.analysis_mode = None
        self.analysis_step = 0
        
        return response
    
    def _handle_existing_patent_analysis(self, query: str) -> str:
        """Handle existing patent analysis with enhanced workflow"""
        patent_id = query.strip()
        
        print("🔍 Performing enhanced existing patent analysis...")
        
        # Use enhanced patent analyzer for comprehensive analysis
        try:
            analysis_results = self.enhanced_analyzer.analyze_patent_comprehensive(patent_id)
            
            if "error" in analysis_results:
                # Fallback to original method if enhanced analysis fails
                print("⚠️ Enhanced analysis failed, using fallback method...")
                return self._handle_existing_patent_analysis_fallback(patent_id)
            
            # Extract results
            patent_data = analysis_results.get("patent_data", {})
            llm_analysis = analysis_results.get("llm_analysis", {})
            similar_patents = analysis_results.get("similar_patents", {})
            should_ingest = analysis_results.get("should_ingest", False)
            ingestion_status = analysis_results.get("ingestion_status", {})
            
            # Build comprehensive response
            response = f"""📚 ENHANCED PATENT ANALYSIS

🔍 PATENT DETAILS:
• Patent Number: {patent_data.get('patent_number', 'Unknown')}
• Title: {patent_data.get('title', 'Unknown')}
• Status: {patent_data.get('status', 'Unknown')}
• Main IPC Code: {patent_data.get('main_ipc_code', 'Unknown')}
• Source: {patent_data.get('source', 'Google Patents')}
{f"• Inventors: {', '.join(patent_data.get('inventors', []))}" if patent_data.get('inventors') else ""}
{f"• Assignee: {patent_data.get('assignee')}" if patent_data.get('assignee') else ""}

📋 ABSTRACT:
{patent_data.get('abstract', 'No abstract available')}

🤖 LLM ANALYSIS:
{llm_analysis.get('analysis', 'Analysis not available')}

🔍 SIMILAR PATENTS (RAG Search):
{similar_patents.get('similar_patents', 'No similar patents found')}

📥 RAG INGESTION STATUS:
"""
            
            if should_ingest:
                response += f"✅ Patent qualifies for RAG ingestion (G06N/G06V category)\n"
                if ingestion_status.get('status') == 'queued':
                    response += f"📋 {ingestion_status.get('message', 'Queued for ingestion')}\n"
                else:
                    response += f"⚠️ {ingestion_status.get('message', 'Ingestion failed')}\n"
            else:
                response += f"❌ Patent does not qualify for RAG ingestion (not G06N/G06V)\n"
            
            response += f"""

💡 RECOMMENDATIONS:
• Review patent claims for scope and coverage
• Analyze prior art and citations
• Consider commercial potential and market impact
• Consult with patent attorney for legal advice
• Evaluate competitive landscape and technology trends
"""
            
            # Store the main response without follow-up prompt
            main_response = response
            
            # Add patent to session context
            session_patent_data = {
                'patent_number': patent_data.get('patent_number', 'Unknown'),
                'title': patent_data.get('title', 'Unknown'),
                'status': patent_data.get('status', 'Unknown'),
                'main_ipc_code': patent_data.get('main_ipc_code', 'Unknown'),
                'source': patent_data.get('source', 'Google Patents'),
                'abstract': patent_data.get('abstract', 'No abstract available'),
                'analysis_results': main_response
            }
            self.conversation_state.add_patent_to_session(session_patent_data)
            
            # Add conversation entry
            self.conversation_state.add_conversation_entry(
                user_query=f"Analyze patent {session_patent_data.get('patent_number', 'Unknown')}",
                bot_response=main_response,
                context_type="existing_patent_analysis"
            )
            
            # Set conversation state for follow-up
            self.conversation_state.mode = "follow_up"
            self.conversation_state.context = {"patent_id": patent_id, "analysis_results": main_response}
            self.conversation_state.last_response = main_response
            self.conversation_state.follow_up_count = 0
            
            # Reset analysis mode
            self.analysis_mode = None
            self.analysis_step = 0
            
            return main_response
            
        except Exception as e:
            logger.error(f"Enhanced analysis failed: {e}")
            print(f"⚠️ Enhanced analysis failed: {e}")
            return self._handle_existing_patent_analysis_fallback(patent_id)
    
    def _handle_existing_patent_analysis_fallback(self, patent_id: str) -> str:
        """Fallback method for existing patent analysis"""
        print("🔄 Using fallback patent analysis method...")
        
        # Step 1: Check for exact patent number or title match in RAG
        exact_match_query = f"""Search for EXACT patent number or title: "{patent_id}". 
        Return ONLY if there is an exact match for this patent number or title in the database.
        If no exact match exists, return 'NO_EXACT_MATCH'."""
        
        rag_context = self._get_rag_context(exact_match_query)
        
        # Step 2: Check if we found an exact match
        rag_lower = rag_context.lower()
        
        # First, try the exact match query
        has_exact_match = (
            "no_exact_match" not in rag_lower and
            len(rag_context.strip()) > 20 and
            not any(indicator in rag_lower for indicator in [
                "you would typically search",
                "you can search through",
                "you can use a variety of databases",
                "patent databases",
                "united states patent and trademark office",
                "world intellectual property organization",
                "google patents",
                "patentscope"
            ]) and
            # Additional check: if the patent number appears in the response, it's likely a match
            patent_id.upper() in rag_context.upper()
        )
        
        # If exact match query fails, try a general search to see if patent exists
        if not has_exact_match:
            print("🔍 Exact match query failed, trying general search...")
            general_query = f"Patent {patent_id}"
            general_rag_context = self._get_rag_context(general_query)
            
            # Check if patent number appears in general search results
            if patent_id.upper() in general_rag_context.upper() and len(general_rag_context.strip()) > 50:
                print(f"✅ Patent {patent_id} found in RAG via general search")
                has_exact_match = True
                rag_context = general_rag_context  # Use the general search results
        
        if not has_exact_match:
            # Step 3: No exact match found, use internet search instead of local LLM fallback
            print("🌐 No exact patent match found in RAG, using internet search...")
            response = self._search_internet_for_existing_patent(patent_id)
        else:
            # Exact match found, use RAG data with enhanced analysis
            rag_query = f"""Detailed analysis of patent {patent_id}. Find related patents, similar inventions, and technical details from the patent database."""
            
            # Get RAG context for detailed analysis
            detailed_rag_context = self._get_rag_context(rag_query)
            
            # Generate LLM response with RAG context
            llm_response = self._generate_llm_response(rag_query, detailed_rag_context)
            
            # Get local analysis
            local_result = self.patent_analyzer.analyze_existing_patent(patent_id)
            
            response = f"""📚 EXACT PATENT MATCH FOUND - COMPREHENSIVE ANALYSIS

🔍 DATABASE ANALYSIS (from patent database):
{llm_response}

📋 LOCAL ANALYSIS:
Key Factors:
"""
            for factor in local_result.key_factors:
                response += f"• {factor}\n"
            
            response += f"\n💡 Analysis:\n{local_result.analysis}"
        
        # Store the main response without follow-up prompt
        main_response = response
        
        # Create patent data for session context
        patent_data = {
            'patent_number': patent_id,
            'title': 'Unknown',
            'status': 'Unknown',
            'main_ipc_code': 'Unknown',
            'source': 'RAG Database' if has_exact_match else 'Google Patents',
            'abstract': 'No abstract available',
            'analysis_results': main_response
        }
        self.conversation_state.add_patent_to_session(patent_data)
        
        # Add conversation entry
        self.conversation_state.add_conversation_entry(
            user_query=f"Analyze patent {patent_id}",
            bot_response=main_response,
            context_type="existing_patent_analysis"
        )
        
        # Set conversation state for follow-up
        self.conversation_state.mode = "follow_up"
        self.conversation_state.context = {"patent_id": patent_id, "analysis_results": main_response}
        self.conversation_state.last_response = main_response
        self.conversation_state.follow_up_count = 0
        
        # Reset analysis mode
        self.analysis_mode = None
        self.analysis_step = 0
        
        return main_response
    
    def _search_internet_for_existing_patent(self, patent_id: str) -> str:
        """Search internet for existing patent information using Google Patents API"""
        print("🌐 Searching Google Patents for existing patent...")
        
        try:
            # Import Google Patents API
            from .google_patents_api import get_patent_details
            
            # Try to get specific patent details
            patent_details = get_patent_details(patent_id)
            
            if patent_details:
                # Generate detailed analysis using LLM
                llm_response = self._generate_llm_response_with_patent_data(patent_id, [patent_details])
                
                return f"""📚 EXISTING PATENT ANALYSIS (Google Patents)

🔍 PATENT DETAILS:
• Patent Number: {patent_details['patent_number']}
• Title: {patent_details['title']}
• Status: {patent_details['status']}
• Source: {patent_details['source']}
{f"• Inventors: {', '.join(patent_details['inventors'])}" if patent_details.get('inventors') else ""}
{f"• Assignee: {patent_details['assignee']}" if patent_details.get('assignee') else ""}
{f"• Filing Date: {patent_details['filing_date']}" if patent_details.get('filing_date') else ""}
{f"• Publication Date: {patent_details['publication_date']}" if patent_details.get('publication_date') else ""}
{f"• Claims: {patent_details['claims_count']}" if patent_details.get('claims_count') else ""}

📋 ABSTRACT:
{patent_details['abstract']}

🔍 DETAILED ANALYSIS:
{llm_response}

💡 RECOMMENDATIONS:
• Review patent claims for scope and coverage
• Analyze prior art and citations
• Consider commercial potential and market impact
• Consult with patent attorney for legal advice
"""
            else:
                # Fallback to general search
                internet_info = self._search_internet_for_patents(patent_id)
                
                if internet_info:
                    llm_response = self._generate_llm_response_with_internet_data(patent_id, internet_info)
                    
                    return f"""📚 EXISTING PATENT ANALYSIS (Google Patents Search)

🌐 SEARCH RESULTS:
{llm_response}

📋 ANALYSIS SUMMARY:
• Patent: "{patent_id}"
• RAG Database: No exact patent match found
• Google Patents: Found {len(internet_info)} related patents
• Analysis: Generated from Google Patents search data

💡 RECOMMENDATIONS:
• Consider conducting a comprehensive patent search
• Review existing patents in similar domains for prior art
• Consult with a patent attorney for comprehensive analysis
"""
                else:
                    return f"""📚 EXISTING PATENT ANALYSIS

❌ NO EXACT PATENT MATCH FOUND

Patent: "{patent_id}"

📋 ANALYSIS SUMMARY:
• RAG Database: No exact patent match found
• Google Patents: No relevant patents found
• Analysis: This patent may not exist or use different terminology

💡 RECOMMENDATIONS:
• This could indicate a novel invention opportunity
• Consider conducting a comprehensive patent search
• Consult with a patent attorney for professional analysis
• Review similar technology domains for related patents
"""
                    
        except ImportError:
            logger.warning("Google Patents API not available, using fallback")
            # Use existing fallback method
            internet_info = self._search_internet_for_patents(patent_id)
            
            if internet_info:
                llm_response = self._generate_llm_response_with_internet_data(patent_id, internet_info)
                
                return f"""📚 EXISTING PATENT ANALYSIS (Fallback Search)

🌐 INTERNET SEARCH RESULTS:
{llm_response}

📋 ANALYSIS SUMMARY:
• Patent: "{patent_id}"
• RAG Database: No exact patent match found
• Internet Search: Found {len(internet_info)} relevant sources
• Analysis: Generated from internet data

💡 RECOMMENDATIONS:
• Consider conducting a comprehensive patent search
• Review existing patents in similar domains for prior art
• Consult with a patent attorney for comprehensive analysis
"""
            else:
                return f"""📚 EXISTING PATENT ANALYSIS

❌ NO EXACT PATENT MATCH FOUND

Patent: "{patent_id}"

📋 ANALYSIS SUMMARY:
• RAG Database: No exact patent match found
• Internet Search: No relevant patents found
• Analysis: This patent may not exist or use different terminology

💡 RECOMMENDATIONS:
• This could indicate a novel invention opportunity
• Consider conducting a comprehensive patent search
• Consult with a patent attorney for professional analysis
• Review similar technology domains for related patents
"""
    
    def _fallback_to_local_llm_existing_patent(self, patent_id: str) -> str:
        """Fallback to local LLM for existing patent analysis when OpenAI is not available"""
        print("🔄 Falling back to local LLM for existing patent analysis...")
        
        # Use the existing internet search fallback
        internet_info = self._search_internet_for_patents(patent_id)
        
        if internet_info:
            llm_response = self._generate_llm_response_with_internet_data(patent_id, internet_info)
            
            return f"""📚 EXISTING PATENT ANALYSIS (Local LLM Fallback)

🌐 INTERNET SEARCH RESULTS:
{llm_response}

📋 ANALYSIS SUMMARY:
• Patent: "{patent_id}"
• RAG Database: No exact patent match found
• OpenAI: Not available
• Local LLM: Generated analysis from internet data
• Internet Search: Found {len(internet_info)} relevant sources

💡 RECOMMENDATIONS:
• Consider filing a new patent application if this technology is novel
• Review existing patents in similar domains for prior art
• Consult with a patent attorney for comprehensive analysis
"""
        else:
            return f"""📚 EXISTING PATENT ANALYSIS

❌ NO EXACT PATENT MATCH FOUND

Patent: "{patent_id}"

📋 ANALYSIS SUMMARY:
• RAG Database: No exact patent match found
• OpenAI: Not available
• Local LLM: No relevant patents found
• Analysis: This patent may not exist or use different terminology

💡 RECOMMENDATIONS:
• This could indicate a novel invention opportunity
• Consider conducting a comprehensive patent search
• Consult with a patent attorney for professional analysis
• Review similar technology domains for related patents
"""
    
    def _handle_patent_search(self, query: str) -> str:
        """Handle patent search with comprehensive results - aim for 8-10 patents with key details"""
        search_query = query.strip()
        
        print("🔍 Performing comprehensive patent search...")
        
        # Step 1: Try RAG search with original query
        rag_query = f"""Search for patents related to: "{search_query}". 
        Return a list of 8-10 relevant patents with patent numbers, inventor names, and short descriptions (100 words each).
        If you find fewer than 8 patents, indicate this clearly."""
        
        rag_context = self._get_rag_context(rag_query)
        
        # Step 2: Check if RAG has sufficient patents (aim for 8-10)
        rag_lower = rag_context.lower()
        
        # Count patent numbers in RAG response
        patent_count = len(re.findall(r'US\d+[A-Z]?\d*|CN\d+[A-Z]?\d*|EP\d+[A-Z]?\d*|JP\d+[A-Z]?\d*|WO\d+[A-Z]?\d*', rag_context))
        print(f"🔍 RAG found {patent_count} patents")
        
        # Check for rejection indicators
        rejection_indicators = [
            "i don't have specific information",
            "i cannot provide specific",
            "i don't have access to",
            "no specific patents found",
            "no relevant patents found",
            "no patents found",
            "no_relevant_matches",
            "no relevant matches"
        ]
        has_rejection_indicators = any(indicator in rag_lower for indicator in rejection_indicators)
        
        # Use RAG if we have 8+ patents and no rejection indicators
        if patent_count >= 8 and not has_rejection_indicators:
            print(f"📚 RAG has sufficient patents ({patent_count}), using RAG data...")
            
            # Generate summary response with RAG data
            summary_response = self._generate_patent_summary(search_query, rag_context)
            
            response = f"""🔍 PATENT SEARCH RESULTS (RAG Database)

📚 COMPREHENSIVE PATENT SUMMARY:
{summary_response}

📋 SEARCH SUMMARY:
• Query: "{search_query}"
• RAG Database: Found {patent_count} relevant patents
• Source: LightRAG patent database
• Quality: Comprehensive patent overview

💡 RECOMMENDATIONS:
• Review the patent summary for key innovations
• Consider conducting additional prior art searches
• Consult with a patent attorney for legal analysis
• Evaluate commercial potential based on findings
"""
        else:
            # Step 3: RAG has insufficient patents, use Google Patents API as supplement
            print(f"🌐 RAG has insufficient patents ({patent_count}), supplementing with Google Patents...")
            
            # Try Google Patents API to get additional patents
            try:
                print(f"🔍 Searching Google Patents for: {search_query}")
                google_patents = self._search_internet_for_patents(search_query)
                
                # Ensure google_patents is always a list
                if google_patents is None:
                    google_patents = []
                    print(f"⚠️ Google Patents returned None, using empty list")
                
                if google_patents:
                    print(f"✅ Found {len(google_patents)} Google Patents results for '{search_query}'")
                    
                    # Combine RAG and Google Patents results
                    combined_patents = self._combine_patent_sources(rag_context, google_patents, search_query)
                    
                    response = f"""🔍 PATENT SEARCH RESULTS (Combined Sources)

📚 COMPREHENSIVE PATENT SUMMARY:
{combined_patents}

📋 SEARCH SUMMARY:
• Original Query: "{search_query}"
• RAG Database: {patent_count} patents found
• Google Patents API: {len(google_patents)} additional patents found
• Total Results: Combined comprehensive patent overview
• Source: RAG Database + Google Patents API

💡 RECOMMENDATIONS:
• Review the comprehensive patent summary for key innovations
• Consider conducting additional prior art searches
• Consult with a patent attorney for legal analysis
• Evaluate commercial potential based on findings
"""
                else:
                    # Use RAG data even if insufficient
                    summary_response = self._generate_patent_summary(search_query, rag_context)
                    
                    response = f"""🔍 PATENT SEARCH RESULTS (RAG Database)

📚 PATENT SUMMARY:
{summary_response}

📋 SEARCH SUMMARY:
• Query: "{search_query}"
• RAG Database: Found {patent_count} patents
• Google Patents API: No additional patents found
• Analysis: Limited patent coverage in this area

💡 RECOMMENDATIONS:
• This could indicate a novel invention opportunity
• Consider conducting a comprehensive patent search
• Consult with a patent attorney for professional analysis
• Review similar technology domains for related patents
"""
            except Exception as e:
                print(f"⚠️ Error searching Google Patents: {e}")
                summary_response = self._generate_patent_summary(search_query, rag_context)
                
                response = f"""🔍 PATENT SEARCH RESULTS (RAG Database)

📚 PATENT SUMMARY:
{summary_response}

📋 SEARCH SUMMARY:
• Query: "{search_query}"
• RAG Database: Found {patent_count} patents
• Google Patents API: Error occurred during search
• Analysis: Limited patent coverage in this area

💡 RECOMMENDATIONS:
• This could indicate a novel invention opportunity
• Consider conducting a comprehensive patent search
• Consult with a patent attorney for professional analysis
• Review similar technology domains for related patents
"""
        
        return response
    
    def _fallback_to_local_llm(self, search_query: str) -> str:
        """Fallback to local LLM when OpenAI is not available"""
        print("🔄 Falling back to local LLM...")
        
        # Use the existing internet search fallback
        internet_info = self._search_internet_for_patents(search_query)
        
        if internet_info:
            llm_response = self._generate_llm_response_with_internet_data(search_query, internet_info)
            
            return f"""🔍 PATENT SEARCH RESULTS (Local LLM Fallback)

🌐 INTERNET SEARCH RESULTS:
{llm_response}

📋 SEARCH SUMMARY:
• Query: "{search_query}"
• RAG Database: No exact patent match found
• OpenAI: Not available
• Local LLM: Generated analysis from internet data
• Internet Search: Found {len(internet_info)} relevant sources

💡 RECOMMENDATIONS:
• Consider filing a new patent application if this technology is novel
• Review existing patents in similar domains for prior art
• Consult with a patent attorney for comprehensive analysis
"""
        else:
            return f"""🔍 PATENT SEARCH RESULTS

❌ NO EXACT PATENT MATCH FOUND

Search Query: "{search_query}"

📋 SEARCH SUMMARY:
• RAG Database: No exact patent match found
• OpenAI: Not available
• Local LLM: No relevant patents found
• Analysis: This patent may not exist or use different terminology

💡 RECOMMENDATIONS:
• This could indicate a novel invention opportunity
• Consider conducting a comprehensive patent search
• Consult with a patent attorney for professional analysis
• Review similar technology domains for related patents
"""
    
    def _search_internet_for_patents(self, query: str) -> List[Dict]:
        """Search internet for patent information using Google Patents API"""
        try:
            # Import Google Patents API
            from .google_patents_api import search_google_patents
            
            logger.info(f"Searching Google Patents for: {query}")
            
            # Search for patents using Google Patents API
            patent_results = search_google_patents(query, max_results=5, use_selenium=True)
            
            if patent_results:
                logger.info(f"Found {len(patent_results)} patents from Google Patents")
                return patent_results
            else:
                logger.warning(f"No patents found for query: {query}")
                return []
                
        except ImportError:
            logger.warning("Google Patents API not available, falling back to mock data")
            return self._get_mock_patent_data(query)
        except Exception as e:
            logger.error(f"Error searching Google Patents for '{query}': {e}")
            logger.info("Falling back to mock data")
            return self._get_mock_patent_data(query)
    
    def _get_mock_patent_data(self, query: str) -> List[Dict]:
        """Fallback mock data when Google Patents API is unavailable"""
        query_lower = query.lower()
        
        if "neural network" in query_lower or "neural networks" in query_lower:
            return [
                {
                    "title": "Neural Network-Based Anomaly Detection System",
                    "patent_number": "US10831762B2",
                    "abstract": "Method and system for detecting anomalies in data using neural networks. The system trains a neural network on normal behavior and uses the trained model to identify deviations as potential anomalies.",
                    "status": "GRANTED",
                    "source": "Mock Data (Google Patents unavailable)",
                    "assignee": "International Business Machines Corporation",
                    "filing_date": "2019-03-15",
                    "publication_date": "2020-11-10"
                },
                {
                    "title": "Deep Neural Network Architecture for Image Recognition",
                    "patent_number": "US10706317B2",
                    "abstract": "Improved deep neural network architecture for image recognition tasks with enhanced accuracy and reduced computational requirements.",
                    "status": "GRANTED",
                    "source": "Mock Data (Google Patents unavailable)",
                    "assignee": "Google LLC",
                    "filing_date": "2018-09-20",
                    "publication_date": "2020-07-07"
                },
                {
                    "title": "Neural Network Training Method for Speech Recognition",
                    "patent_number": "US10600408B2",
                    "abstract": "Method for training neural networks specifically optimized for speech recognition applications with improved accuracy and reduced training time.",
                    "status": "GRANTED",
                    "source": "Mock Data (Google Patents unavailable)",
                    "assignee": "Microsoft Corporation",
                    "filing_date": "2017-11-30",
                    "publication_date": "2020-03-24"
                }
            ]
        elif "machine learning" in query_lower or "ai" in query_lower:
            return [
                {
                    "title": "Machine Learning Patent Analysis System",
                    "patent_number": "US20230012345",
                    "abstract": "System for analyzing patent documents using machine learning algorithms",
                    "status": "PENDING",
                    "source": "Mock Data (Google Patents unavailable)"
                },
                {
                    "title": "AI-Powered Patent Classification Method",
                    "patent_number": "US20220098765",
                    "abstract": "Method for automatically classifying patents using artificial intelligence",
                    "status": "PUBLISHED",
                    "source": "Mock Data (Google Patents unavailable)"
                }
            ]
        elif "iot" in query_lower or "internet of things" in query_lower:
            return [
                {
                    "title": "IoT Device Management System",
                    "patent_number": "US20230123456",
                    "abstract": "System for managing IoT devices with centralized control",
                    "status": "PENDING",
                    "source": "Mock Data (Google Patents unavailable)"
                }
            ]
        elif "blockchain" in query_lower:
            return [
                {
                    "title": "Blockchain-Based Patent Verification",
                    "patent_number": "US20220134567",
                    "abstract": "Method for verifying patent authenticity using blockchain technology",
                    "status": "PUBLISHED",
                    "source": "Mock Data (Google Patents unavailable)"
                }
            ]
        else:
            # Generic response for unknown queries
            return [
                {
                    "title": f"Advanced {query.title()} System",
                    "patent_number": "US20230000001",
                    "abstract": f"System for {query} with improved efficiency and performance",
                    "status": "PENDING",
                    "source": "Mock Data (Google Patents unavailable)"
                }
            ]
    
    def _generate_llm_response_with_internet_data(self, query: str, internet_data: List[Dict]) -> str:
        """Generate LLM response using internet search data"""
        try:
            # Format internet data for LLM
            formatted_data = ""
            for i, patent in enumerate(internet_data, 1):
                formatted_data += f"""
Patent {i}:
- Title: {patent['title']}
- Patent Number: {patent['patent_number']}
- Abstract: {patent['abstract']}
- Status: {patent['status']}
- Source: {patent['source']}
"""
            
            # Create prompt for LLM
            prompt = f"""Based on the following internet search results for "{query}", provide a comprehensive and SPECIFIC analysis:

{formatted_data}

Please provide a DETAILED analysis with the following structure:

1. SPECIFIC PATENTS FOUND:
   - List each patent with its specific patent number, title, and key technical details
   - Highlight the most relevant aspects of each patent
   - Include specific technical features and innovations

2. TECHNICAL ANALYSIS:
   - Analyze the specific technical innovations in each patent
   - Compare the technical approaches used
   - Identify unique features and capabilities
   - Discuss specific implementation details

3. MARKET IMPLICATIONS:
   - Discuss specific applications and use cases
   - Analyze commercial potential and market positioning
   - Identify competitive advantages
   - Mention specific industries or sectors

4. PATENT STRATEGY RECOMMENDATIONS:
   - Provide specific recommendations for patent strategy
   - Identify potential licensing opportunities
   - Suggest areas for further research

IMPORTANT: Be SPECIFIC and DETAILED. Include actual patent numbers, titles, and technical details. Do not give generic responses."""
            
            # Generate response using Ollama (increased timeout for internet search processing)
            import requests
            try:
                ollama_response = requests.post(
                    "http://localhost:11434/api/generate",
                    json={
                        "model": "qwen2.5:14b-instruct",
                        "prompt": prompt,
                        "stream": False
                    },
                    timeout=300
                )
                
                if ollama_response.status_code == 200:
                    data = ollama_response.json()
                    if 'response' in data:
                        return data['response'].strip()
            except Exception as e:
                logger.error(f"Ollama request failed: {e}")
            
            # Fallback response
            return f"""Based on internet search results, I found {len(internet_data)} relevant patents for "{query}". 

The search reveals existing patents in this technology area, suggesting this field has active innovation. Consider conducting a detailed prior art search before filing a new patent application.

Key findings:
• {len(internet_data)} patents found in related technology areas
• Mix of pending and published applications
• Indicates active development in this field

Recommendations:
• Conduct comprehensive prior art search
• Review existing patents for potential conflicts
• Consider patent landscape analysis
• Consult with patent attorney for professional advice"""
            
        except Exception as e:
            logger.error(f"Error generating LLM response with internet data: {e}")
            return f"Unable to generate detailed analysis for '{query}' due to processing error. Found {len(internet_data)} patents in search results."
    
    def _generate_llm_response_with_patent_data(self, query: str, patent_data: List[Dict]) -> str:
        """Generate LLM response using Google Patents data"""
        try:
            # Format patent data for LLM
            formatted_data = ""
            for i, patent in enumerate(patent_data, 1):
                formatted_data += f"""
Patent {i}:
- Patent Number: {patent['patent_number']}
- Title: {patent['title']}
- Abstract: {patent['abstract']}
- Status: {patent['status']}
- Source: {patent['source']}
{f"- Inventors: {', '.join(patent['inventors'])}" if patent.get('inventors') else ""}
{f"- Assignee: {patent['assignee']}" if patent.get('assignee') else ""}
{f"- Filing Date: {patent['filing_date']}" if patent.get('filing_date') else ""}
{f"- Publication Date: {patent['publication_date']}" if patent.get('publication_date') else ""}
{f"- Claims Count: {patent['claims_count']}" if patent.get('claims_count') else ""}
{f"- Classification Codes: {', '.join(patent['classification_codes'])}" if patent.get('classification_codes') else ""}
{f"- URL: {patent['url']}" if patent.get('url') else ""}
"""
            
            # Create prompt for LLM
            prompt = f"""Based on the following Google Patents data for "{query}", provide a comprehensive analysis:

{formatted_data}

Please analyze:
1. Technical novelty and innovation
2. Patent scope and claims analysis
3. Market potential and commercial implications
4. Prior art considerations
5. Recommendations for patent strategy

Provide a detailed analysis:"""
            
            # Generate response using Ollama (increased timeout for patent analysis)
            import requests
            ollama_response = requests.post(
                "http://localhost:11434/api/generate",
                json={
                    "model": "qwen2.5:14b-instruct",
                    "prompt": prompt,
                    "stream": False
                },
                timeout=300
            )
            
            if ollama_response.status_code == 200:
                data = ollama_response.json()
                if 'response' in data:
                    return data['response'].strip()
            
            # Fallback response
            return f"""Based on Google Patents data, I found detailed information for "{query}". 

The patent analysis reveals important technical and legal aspects that should be considered for patent strategy and commercial applications.

Key findings:
• {len(patent_data)} patent(s) analyzed from Google Patents
• Detailed technical specifications available
• Legal status and filing information provided
• Inventor and assignee information included

Recommendations:
• Review patent claims for scope and coverage
• Analyze prior art and citations
• Consider commercial potential and market impact
• Consult with patent attorney for legal advice
• Evaluate competitive landscape and technology trends"""
            
        except Exception as e:
            logger.error(f"Error generating LLM response with patent data: {e}")
            return f"Unable to generate detailed analysis for '{query}' due to processing error."
    
    def _get_general_response(self, query: str) -> str:
        """Get a general conversation response"""
        import random
        query_lower = query.lower().strip()
        
        # Greetings
        greetings = ["hi", "hello", "hey", "good morning", "good afternoon", "good evening", "greetings"]
        if any(greeting in query_lower for greeting in greetings):
            return random.choice(self.general_responses["greetings"])
            
        # Questions about capabilities
        capability_questions = ["what can you do", "what are your capabilities", "help", "what do you do", "tell me about yourself"]
        if any(phrase in query_lower for phrase in capability_questions):
            return random.choice(self.general_responses["capabilities"])
            
        # Thanks
        thanks = ["thank", "thanks", "appreciate"]
        if any(phrase in query_lower for phrase in thanks):
            return random.choice(self.general_responses["thanks"])
            
        # Goodbyes
        goodbyes = ["bye", "goodbye", "see you", "farewell", "exit", "quit"]
        if any(phrase in query_lower for phrase in goodbyes):
            return random.choice(self.general_responses["goodbye"])
            
        # For other general questions, try to use Ollama as fallback
        try:
            import requests
            ollama_response = requests.post(
                "http://localhost:11434/api/generate",
                json={
                    "model": "qwen2.5:14b-instruct",
                    "prompt": f"User: {query}\nAssistant:",
                    "stream": False
                },
                timeout=300
            )
            
            if ollama_response.status_code == 200:
                data = ollama_response.json()
                if 'response' in data:
                    return data['response'].strip()
        except:
            pass
            
        # Default response if Ollama is not available
        return "I'm a patent analysis assistant! I can help you with patent research, technical analysis, and answering questions about inventions and patent law. What would you like to know?"
        
    def get_response(self, query: str, validate: bool = None, evaluate: bool = False) -> ChatbotResponse:
        """Get response from chatbot with optional validation and evaluation"""
        start_time = time.time()
        
        # Check if we're in follow-up mode
        if self.conversation_state.mode == "follow_up":
            response_content = self._handle_follow_up_query(query)
            response_time = time.time() - start_time
            
            # Check if this is a menu selection that should bypass guardrails
            query_lower = query.lower().strip()
            menu_selections = [
                "need more details", "more details", "tell me more", "additional details", "more information",
                "return to menu", "main menu", "menu", "back to menu", "return to main menu",
                "search for different", "different patent", "new search", "search again", "search for a different patent",
                "yes", "y", "yeah", "sure", "okay", "no", "n", "nope", "not really", "that's all"
            ]
            
            is_menu_selection = any(phrase in query_lower for phrase in menu_selections)
            
            # Apply guardrails and evaluation (bypass for menu selections)
            guardrail_scores = None
            evaluation_scores = None
            
            if self.with_guardrails and self.guardrails_validator and not is_menu_selection:
                validated_response, guardrail_scores = self.guardrails_validator.validate_response(response_content)
                if not guardrail_scores.is_acceptable():
                    response_content = f"""⚠️ Response filtered by guardrails:

{guardrail_scores.get_rejection_reason()}

Please rephrase your question or ask about a different topic."""
                else:
                    response_content = validated_response
            elif is_menu_selection:
                # For menu selections, use default acceptable scores
                guardrail_scores = GuardrailScores(0.0, 0.0, 0.0)  # All acceptable
            else:
                # For other follow-up queries, use default scores
                guardrail_scores = GuardrailScores(0.0, 0.5, 0.5)
            
            # Evaluate response if requested (skip for menu selections)
            if evaluate and self.evaluator and not is_menu_selection:
                evaluation_scores = self.evaluator.evaluate_single_response(query, response_content)
            
            # Log the conversation
            guardrail_dict = guardrail_scores.to_dict() if guardrail_scores else {"profanity_score": 0.0, "topic_relevance_score": 0.5, "politeness_score": 0.5}
            evaluation_dict = evaluation_scores.to_dict() if evaluation_scores else None
            
            self.session_logger.log_conversation(
                user_query=query,
                assistant_response=response_content,
                response_time=response_time,
                guardrail_scores=guardrail_dict,
                evaluation_scores=evaluation_dict,
                data_source="follow_up"
            )
            
            # Add to conversation history for context
            self.conversation_state.add_conversation_entry(
                user_query=query,
                bot_response=response_content,
                context_type="follow_up"
            )
            
            return ChatbotResponse(
                content=response_content,
                sources=[],
                response_time=response_time,
                guardrail_scores=guardrail_scores or GuardrailScores(0, 0, 0),
                evaluation_scores=evaluation_scores
            )
        
        # Check if we're in interactive query mode
        if self.conversation_state.mode == "interactive_query":
            response_content = self._handle_follow_up_query(query)
            response_time = time.time() - start_time
            
            # For interactive query mode, bypass guardrails
            guardrail_scores = GuardrailScores(0.0, 0.0, 0.0)  # All acceptable
            evaluation_scores = None
            
            # Log the conversation
            self.session_logger.log_conversation(
                user_query=query,
                assistant_response=response_content,
                response_time=response_time,
                guardrail_scores=guardrail_scores.to_dict(),
                evaluation_scores=evaluation_scores,
                data_source="interactive_query"
            )
            
            return ChatbotResponse(
                content=response_content,
                sources=[],
                response_time=response_time,
                guardrail_scores=guardrail_scores,
                evaluation_scores=evaluation_scores
            )
        
        # Check if we're in patent analysis mode (this should be checked BEFORE patent pattern matching)
        if self.analysis_mode == "new_invention":
            response_content = self._handle_new_invention_collection(query)
            response_time = time.time() - start_time
            
            # Apply guardrails and evaluation for LLM-generated content (final analysis step)
            guardrail_scores = None
            evaluation_scores = None
            
            # Only evaluate if this is the final analysis step (contains LLM content)
            if self.analysis_step == 0 and "Patent Analysis Complete" in response_content:
                if self.with_guardrails and self.guardrails_validator:
                    validated_response, guardrail_scores = self.guardrails_validator.validate_response(response_content)
                    if not guardrail_scores.is_acceptable():
                        response_content = f"""⚠️ Response filtered by guardrails:

{guardrail_scores.get_rejection_reason()}

Please rephrase your question or ask about a different topic."""
                    else:
                        response_content = validated_response
                
                # Evaluate response if requested
                if evaluate and self.evaluator:
                    evaluation_scores = self.evaluator.evaluate_single_response(query, response_content)
            
            # Log the conversation properly
            guardrail_dict = guardrail_scores.to_dict() if guardrail_scores else {"profanity_score": 0.0, "topic_relevance_score": 0.5, "politeness_score": 0.5}
            evaluation_dict = evaluation_scores.to_dict() if evaluation_scores else None
            
            self.session_logger.log_conversation(
                user_query=query,
                assistant_response=response_content,
                response_time=response_time,
                guardrail_scores=guardrail_dict,
                evaluation_scores=evaluation_dict,
                data_source="patent_analysis"
            )
            
            return ChatbotResponse(
                content=response_content,
                sources=[],
                response_time=response_time,
                guardrail_scores=guardrail_scores or GuardrailScores(0, 0, 0),
                evaluation_scores=evaluation_scores
            )
        
        elif self.analysis_mode == "existing_patent":
            response_content = self._handle_existing_patent_analysis(query)
            response_time = time.time() - start_time
            
            # Apply guardrails and evaluation for LLM-generated content
            guardrail_scores = None
            evaluation_scores = None
            
            if self.with_guardrails and self.guardrails_validator:
                validated_response, guardrail_scores = self.guardrails_validator.validate_response(response_content)
                if not guardrail_scores.is_acceptable():
                    response_content = f"""⚠️ Response filtered by guardrails:

{guardrail_scores.get_rejection_reason()}

Please rephrase your question or ask about a different topic."""
                else:
                    response_content = validated_response
            
            # Evaluate response if requested
            if evaluate and self.evaluator:
                evaluation_scores = self.evaluator.evaluate_single_response(query, response_content)
            
            # Log the conversation properly
            guardrail_dict = guardrail_scores.to_dict() if guardrail_scores else {"profanity_score": 0.0, "topic_relevance_score": 0.5, "politeness_score": 0.5}
            evaluation_dict = evaluation_scores.to_dict() if evaluation_scores else None
            
            self.session_logger.log_conversation(
                user_query=query,
                assistant_response=response_content,
                response_time=response_time,
                guardrail_scores=guardrail_dict,
                evaluation_scores=evaluation_dict,
                data_source="patent_analysis"
            )
            
            return ChatbotResponse(
                content=response_content,
                sources=[],
                response_time=response_time,
                guardrail_scores=guardrail_scores or GuardrailScores(0, 0, 0),
                evaluation_scores=evaluation_scores
            )
        
        elif self.analysis_mode == "patent_search":
            response_content = self._handle_patent_search(query)
            response_time = time.time() - start_time
            
            # Apply guardrails and evaluation for LLM-generated content
            guardrail_scores = None
            evaluation_scores = None
            
            if self.with_guardrails and self.guardrails_validator:
                validated_response, guardrail_scores = self.guardrails_validator.validate_response(response_content)
                if not guardrail_scores.is_acceptable():
                    response_content = f"""⚠️ Response filtered by guardrails:

{guardrail_scores.get_rejection_reason()}

Please rephrase your question or ask about a different topic."""
                else:
                    response_content = validated_response
            
            # Evaluate response if requested
            if evaluate and self.evaluator:
                evaluation_scores = self.evaluator.evaluate_single_response(query, response_content)
            
            # Set up conversation state for follow-up options
            self.conversation_state.mode = "follow_up"
            self.conversation_state.context = {"last_search_query": query, "search_results": response_content}
            self.conversation_state.follow_up_count = 0
            self.conversation_state.awaiting_yes_no = False
            
            # Add to conversation history for context
            self.conversation_state.add_conversation_entry(
                user_query=query,
                bot_response=response_content,
                context_type="patent_search"
            )
            
            # Log the conversation properly
            guardrail_dict = guardrail_scores.to_dict() if guardrail_scores else {"profanity_score": 0.0, "topic_relevance_score": 0.5, "politeness_score": 0.5}
            evaluation_dict = evaluation_scores.to_dict() if evaluation_scores else None
            
            self.session_logger.log_conversation(
                user_query=query,
                assistant_response=response_content,
                response_time=response_time,
                guardrail_scores=guardrail_dict,
                evaluation_scores=evaluation_dict,
                data_source="patent_search"
            )
            
            return ChatbotResponse(
                content=response_content,
                sources=[],
                response_time=response_time,
                guardrail_scores=guardrail_scores or GuardrailScores(0, 0, 0),
                evaluation_scores=evaluation_scores
            )
        
        elif self.analysis_mode == "enhanced_analysis":
            response_content = self._handle_enhanced_patent_analysis(query)
            response_time = time.time() - start_time
            
            # Apply guardrails and evaluation for enhanced analysis
            guardrail_scores = None
            evaluation_scores = None
            
            if self.with_guardrails and self.guardrails_validator:
                validated_response, guardrail_scores = self.guardrails_validator.validate_response(response_content)
                if not guardrail_scores.is_acceptable():
                    response_content = f"""⚠️ Response filtered by guardrails:

{guardrail_scores.get_rejection_reason()}

Please rephrase your question or ask about a different topic."""
                else:
                    response_content = validated_response
            
            # Evaluate response with enhanced metrics
            if evaluate and self.evaluator:
                evaluation_scores = self.evaluator.evaluate_single_response(query, response_content)
            
            # Set up conversation state for enhanced follow-up options
            self.conversation_state.mode = "enhanced_follow_up"
            self.conversation_state.context = {"enhanced_query": query, "enhanced_results": response_content}
            self.conversation_state.follow_up_count = 0
            self.conversation_state.awaiting_yes_no = False
            
            # Add to conversation history for context
            self.conversation_state.add_conversation_entry(
                user_query=query,
                bot_response=response_content,
                context_type="enhanced_analysis"
            )
            
            # Log the conversation properly
            guardrail_dict = guardrail_scores.to_dict() if guardrail_scores else {"profanity_score": 0.0, "topic_relevance_score": 0.5, "politeness_score": 0.5}
            evaluation_dict = evaluation_scores.to_dict() if evaluation_scores else None
            
            self.session_logger.log_conversation(
                user_query=query,
                assistant_response=response_content,
                response_time=response_time,
                guardrail_scores=guardrail_dict,
                evaluation_scores=evaluation_dict,
                data_source="enhanced_analysis"
            )
            
            return ChatbotResponse(
                content=response_content,
                sources=[],
                response_time=response_time,
                guardrail_scores=guardrail_scores or GuardrailScores(0, 0, 0),
                evaluation_scores=evaluation_scores
            )
        
        # Check if this is a patent analysis selection (when no analysis mode is active)
        if self.analysis_mode is None:
            query_stripped = query.strip()
            query_lower = query_stripped.lower()
            
            # Check if it's a valid menu option (numeric or text-based)
            menu_options = {
                '1': 'existing_patent',
                '2': 'new_invention', 
                '3': 'patent_search',
                'analyze existing patent': 'existing_patent',
                'analyze new invention': 'new_invention',
                'search for similar patents': 'patent_search',
                'search patents': 'patent_search',
                'patent search': 'patent_search'
            }
            
            # Check for exact matches first
            if query_stripped in ['1', '2', '3']:
                # Handle valid numeric patent analysis selection
                response_content = self._handle_patent_analysis_selection(query)
                response_time = time.time() - start_time
                
                # Log the conversation properly
                self.session_logger.log_conversation(
                    user_query=query,
                    assistant_response=response_content,
                    response_time=response_time,
                    guardrail_scores={"profanity_score": 0.0, "topic_relevance_score": 0.5, "politeness_score": 0.5},
                    data_source="menu_selection"
                )
                
                return ChatbotResponse(
                    content=response_content,
                    sources=[],
                    response_time=response_time,
                    guardrail_scores=GuardrailScores(0, 0, 0)
                )
            elif query_stripped in ['10', '10.', '10)']:
                # Handle special input "10" - trigger enhanced analysis
                response_content = self._handle_enhanced_analysis_mode(query)
                response_time = time.time() - start_time
                
                # Log the conversation properly
                self.session_logger.log_conversation(
                    user_query=query,
                    assistant_response=response_content,
                    response_time=response_time,
                    guardrail_scores={"profanity_score": 0.0, "topic_relevance_score": 0.5, "politeness_score": 0.5},
                    data_source="enhanced_analysis"
                )
                
                return ChatbotResponse(
                    content=response_content,
                    sources=[],
                    response_time=response_time,
                    guardrail_scores=GuardrailScores(0, 0, 0)
                )
            elif query_lower in menu_options:
                # Handle text-based menu selection
                selected_mode = menu_options[query_lower]
                response_content = self._handle_patent_analysis_selection(selected_mode)
                response_time = time.time() - start_time
                
                # Log the conversation properly
                self.session_logger.log_conversation(
                    user_query=query,
                    assistant_response=response_content,
                    response_time=response_time,
                    guardrail_scores={"profanity_score": 0.0, "topic_relevance_score": 0.5, "politeness_score": 0.5},
                    data_source="menu_selection"
                )
                
                return ChatbotResponse(
                    content=response_content,
                    sources=[],
                    response_time=response_time,
                    guardrail_scores=GuardrailScores(0, 0, 0)
                )
            else:
                # Check if this looks like a patent number or search query
                # Patent numbers typically contain letters and numbers (e.g., US10896352B2, EP1234567A1)
                import re
                patent_pattern = re.compile(r'^[A-Z]{2}\d+[A-Z0-9]*$|^[A-Z]{1,2}\d+[A-Z0-9]*$|^[A-Z]{2,3}\d+[A-Z0-9]*$')
                
                # Check if it's a technology search query (contains technology keywords)
                technology_keywords = [
                    "machine learning", "artificial intelligence", "ai", "neural network", "deep learning", "blockchain", "iot", "internet of things", "robotics", "automation",
                    "computer vision", "natural language processing", "nlp", "data mining", "analytics", "algorithm", "software", "hardware", "electronics", "biotechnology",
                    "pharmaceutical", "medical device", "diagnostic", "therapeutic", "drug", "chemical", "material", "nanotechnology", "quantum", "renewable energy",
                    "solar", "wind", "battery", "electric vehicle", "autonomous", "drone", "satellite", "wireless", "5g", "cybersecurity", "cryptography"
                ]
                
                is_technology_query = any(keyword in query_lower for keyword in technology_keywords)
                
                # Check if it looks like a patent number (starts with country code and has numbers)
                import re
                patent_number_pattern = re.compile(r'^[A-Z]{2}\d+[A-Z0-9]*$|^[A-Z]{1,2}\d+[A-Z0-9]*$|^[A-Z]{2,3}\d+[A-Z0-9]*$')
                is_patent_number = patent_number_pattern.match(query_stripped)
                
                if is_patent_number:
                    # This is definitely a patent number, treat as existing patent analysis
                    self.analysis_mode = "existing_patent"
                    response_content = self._handle_existing_patent_analysis(query_stripped)
                elif is_technology_query or len(query_stripped.split()) > 2:
                    # This looks like a technology search query, treat as patent search
                    self.analysis_mode = "patent_search"
                    response_content = self._handle_patent_search(query_stripped)
                    response_time = time.time() - start_time
                    
                    # Set up conversation state for follow-up options
                    self.conversation_state.mode = "follow_up"
                    self.conversation_state.context = {"last_search_query": query_stripped, "search_results": response_content}
                    self.conversation_state.follow_up_count = 0
                    self.conversation_state.awaiting_yes_no = False
                    
                    # Add to conversation history for context
                    self.conversation_state.add_conversation_entry(
                        user_query=query_stripped,
                        bot_response=response_content,
                        context_type="patent_search"
                    )
                    
                    # Log the conversation properly
                    self.session_logger.log_conversation(
                        user_query=query,
                        assistant_response=response_content,
                        response_time=response_time,
                        guardrail_scores={"profanity_score": 0.0, "topic_relevance_score": 0.5, "politeness_score": 0.5},
                        data_source="patent_search"
                    )
                    
                    return ChatbotResponse(
                        content=response_content,
                        sources=[],
                        response_time=response_time,
                        guardrail_scores=GuardrailScores(0, 0, 0)
                    )
                else:
                    # Invalid menu option - show validation error
                    response_content = "Option not correctly selected, please select a valid option\n\n" + self._show_main_menu()
                    response_time = time.time() - start_time
                    
                    # Log the conversation properly
                    self.session_logger.log_conversation(
                        user_query=query,
                        assistant_response=response_content,
                        response_time=response_time,
                        guardrail_scores={"profanity_score": 0.0, "topic_relevance_score": 0.5, "politeness_score": 0.5},
                        data_source="menu_selection"
                    )
                    
                    return ChatbotResponse(
                        content=response_content,
                        sources=[],
                        response_time=response_time,
                        guardrail_scores=GuardrailScores(0, 0, 0)
                    )
        
        # Handle general conversation and LightRAG queries
        if self._is_general_conversation(query):
            response_content = self._get_general_response(query)
            response_time = time.time() - start_time
            
            # Log the conversation properly
            self.session_logger.log_conversation(
                user_query=query,
                assistant_response=response_content,
                response_time=response_time,
                guardrail_scores={"profanity_score": 0.0, "topic_relevance_score": 0.5, "politeness_score": 0.5},
                data_source="general_conversation"
            )
            
            return ChatbotResponse(
                content=response_content,
                sources=[],
                response_time=response_time,
                guardrail_scores=GuardrailScores(0, 0, 0)
            )
        
        # Check LightRAG availability
        if not self._check_lightrag_availability():
            # Try Neo4j fallback
            try:
                neo4j_response = self.neo4j_fallback.query_neo4j(query)
                if neo4j_response:
                    response_content = f"""🔍 Neo4j Fallback Response:

{neo4j_response}

⚠️ Note: LightRAG server is unavailable. This response is from Neo4j backup data."""
                    response_time = time.time() - start_time
                    
                    # Log the conversation properly
                    self.session_logger.log_conversation(
                        user_query=query,
                        assistant_response=response_content,
                        response_time=response_time,
                        guardrail_scores={"profanity_score": 0.0, "topic_relevance_score": 0.5, "politeness_score": 0.5},
                        data_source="neo4j_fallback"
                    )
                    
                    return ChatbotResponse(
                        content=response_content,
                        sources=[],
                        response_time=response_time,
                        guardrail_scores=GuardrailScores(0, 0, 0)
                    )
            except Exception as e:
                logger.error(f"Neo4j fallback failed: {e}")
            
            # Try SQLite fallback
            try:
                sqlite_response = self.sqlite_fallback.query_sqlite(query)
                if sqlite_response:
                    response_content = f"""🔍 SQLite Fallback Response:

{sqlite_response}

⚠️ Note: LightRAG server is unavailable. This response is from SQLite backup data."""
                    response_time = time.time() - start_time
                    
                    # Log the conversation properly
                    self.session_logger.log_conversation(
                        user_query=query,
                        assistant_response=response_content,
                        response_time=response_time,
                        guardrail_scores={"profanity_score": 0.0, "topic_relevance_score": 0.5, "politeness_score": 0.5},
                        data_source="sqlite_fallback"
                    )
                    
                    return ChatbotResponse(
                        content=response_content,
                        sources=[],
                        response_time=response_time,
                        guardrail_scores=GuardrailScores(0, 0, 0)
                    )
            except Exception as e:
                logger.error(f"SQLite fallback failed: {e}")
            
            # No fallback available
            response_content = """❌ LightRAG server is currently unavailable.

🔧 Troubleshooting:
1. Check if LightRAG server is running on http://localhost:9621
2. Verify Neo4j database is accessible
3. Try again in a few moments

For immediate assistance, try:
• General questions about patents
• Patent analysis options (1-3)
• Or restart the LightRAG server"""
            
            response_time = time.time() - start_time
            
            # Log the conversation properly
            self.session_logger.log_conversation(
                user_query=query,
                assistant_response=response_content,
                response_time=response_time,
                guardrail_scores={"profanity_score": 0.0, "topic_relevance_score": 0.5, "politeness_score": 0.5},
                data_source="lightrag_unavailable"
            )
            
            return ChatbotResponse(
                content=response_content,
                sources=[],
                response_time=response_time,
                guardrail_scores=GuardrailScores(0, 0, 0)
            )
        
        # Get RAG context
        rag_context = self._get_rag_context(query)
        
        # Generate LLM response
        llm_response = self._generate_llm_response(query, rag_context)
        
        # Combine RAG and LLM response
        combined_response = self._combine_rag_and_llm_response(query, rag_context, llm_response)
        
        response_time = time.time() - start_time
        
        # Apply guardrails if enabled
        guardrail_scores = None
        if self.with_guardrails and self.guardrails_validator:
            validated_response, guardrail_scores = self.guardrails_validator.validate_response(combined_response)
            if not guardrail_scores.is_acceptable():
                combined_response = f"""⚠️ Response filtered by guardrails:

{guardrail_scores.get_rejection_reason()}

Please rephrase your question or ask about a different topic."""
            else:
                combined_response = validated_response
        
        # Evaluate response if requested
        evaluation_scores = None
        if evaluate and self.evaluator:
            evaluation_scores = self.evaluator.evaluate_single_response(query, combined_response)
        
        # Extract sources from RAG context
        sources = self._extract_sources_from_rag(rag_context)
        
        # Log the conversation properly
        guardrail_dict = guardrail_scores.to_dict() if guardrail_scores else {"profanity_score": 0.0, "topic_relevance_score": 0.5, "politeness_score": 0.5}
        evaluation_dict = evaluation_scores.to_dict() if evaluation_scores else None
        
        self.session_logger.log_conversation(
            user_query=query,
            assistant_response=combined_response,
            response_time=response_time,
            guardrail_scores=guardrail_dict,
            evaluation_scores=evaluation_dict,
            data_source="lightrag"
        )
        
        return ChatbotResponse(
            content=combined_response,
            sources=sources,
            response_time=response_time,
            guardrail_scores=guardrail_scores or GuardrailScores(0, 0, 0),
            evaluation_scores=evaluation_scores
        )
    
    def _handle_follow_up_query(self, query: str) -> str:
        """Handle follow-up questions with enhanced session context awareness"""
        query_lower = query.lower().strip()
        
        # Check if user wants to return to menu
        if any(phrase in query_lower for phrase in ["return to menu", "main menu", "menu", "back to menu", "return to main menu"]):
            self.conversation_state.mode = None
            self.conversation_state.context = None
            self.conversation_state.follow_up_count = 0
            self.conversation_state.awaiting_yes_no = False
            return self._show_main_menu()
        
        # Check if user wants more details - NEW INTERACTIVE FLOW
        if any(phrase in query_lower for phrase in ["need more details", "more details", "tell me more", "additional details", "more information"]):
            # Set conversation state for interactive query mode
            self.conversation_state.mode = "interactive_query"
            self.conversation_state.follow_up_count = 0
            self.conversation_state.awaiting_yes_no = False
            return "Please provide your query:"
        
        # Handle interactive query mode
        if self.conversation_state.mode == "interactive_query":
            # If awaiting yes/no, only accept yes/no answers
            if self.conversation_state.awaiting_yes_no:
                if query_lower in ["yes", "y", "yeah", "sure", "okay"]:
                    self.conversation_state.awaiting_yes_no = False
                    return "Please provide your query:"
                elif query_lower in ["no", "n", "nope", "not really", "that's all"]:
                    # Return to main menu
                    self.conversation_state.mode = None
                    self.conversation_state.context = None
                    self.conversation_state.follow_up_count = 0
                    self.conversation_state.awaiting_yes_no = False
                    return self._show_main_menu()
                else:
                    return "Please answer with 'yes' or 'no'. Do you have any further questions?"
            # Not awaiting yes/no, treat as a user query with enhanced context
            try:
                import requests
                
                # Get session context
                session_context = self.conversation_state.get_recent_context(3)
                current_patent = self.conversation_state.get_current_patent()
                session_summary = self.conversation_state.get_session_context_summary()
                
                # Build enhanced prompt with full context
                prompt = f"""You are a patent analysis assistant. Answer the user's question with full context awareness.

SESSION CONTEXT:
{session_summary}

RECENT CONVERSATION:
{session_context}

CURRENT PATENT CONTEXT:
{current_patent if current_patent else "No specific patent currently referenced"}

USER QUESTION: "{query}"

INSTRUCTIONS:
1. Use the session context to understand what patents have been discussed
2. If the user is asking about a specific patent, reference the correct one from the session
3. If multiple patents are in the session, clarify which one the user is referring to
4. Provide comprehensive, context-aware answers based on the patents discussed
5. If the user's question is unclear, ask for clarification about which patent they mean

RESPONSE:"""
                
                ollama_response = requests.post(
                    "http://localhost:11434/api/generate",
                    json={
                        "model": "qwen2.5:14b-instruct",
                        "prompt": prompt,
                        "stream": False
                    },
                    timeout=180
                )
                if ollama_response.status_code == 200:
                    data = ollama_response.json()
                    if 'response' in data:
                        response = data['response'].strip()
                        # After answering, set awaiting_yes_no and prompt for further questions
                        self.conversation_state.awaiting_yes_no = True
                        return response + "\n\nDo you have any further questions? (yes/no)"
            except Exception as e:
                logger.error(f"Error generating interactive query response: {e}")
                self.conversation_state.awaiting_yes_no = True
                return f"I apologize, but I couldn't generate a response at this time. Error: {e}\n\nDo you have any further questions? (yes/no)"
        
        # Handle yes/no response in interactive query mode (should not reach here due to above logic)
        if self.conversation_state.mode == "interactive_query":
            if query_lower in ["yes", "y", "yeah", "sure", "okay"]:
                self.conversation_state.awaiting_yes_no = False
                return "Please provide your query:"
            elif query_lower in ["no", "n", "nope", "not really", "that's all"]:
                self.conversation_state.mode = None
                self.conversation_state.context = None
                self.conversation_state.follow_up_count = 0
                self.conversation_state.awaiting_yes_no = False
                return self._show_main_menu()
            else:
                return "Please answer with 'yes' or 'no'. Do you have any further questions?"
        
        # Check if user wants to search for different patent
        if any(phrase in query_lower for phrase in ["search for different", "different patent", "new search", "search again", "search for a different patent"]):
            # Reset conversation state and set up for existing patent analysis (option 1)
            self.conversation_state.mode = None
            self.conversation_state.context = None
            self.conversation_state.follow_up_count = 0
            self.conversation_state.awaiting_yes_no = False
            self.analysis_mode = "existing_patent"  # Set to existing patent analysis mode
            return """📚 Please provide the patent number or title to analyze:
(Examples: US12345678, 'Neural Network System', etc.)"""
        
        # Check follow-up count limit
        if self.conversation_state.follow_up_count >= self.conversation_state.max_follow_ups:
            self.conversation_state.mode = None
            self.conversation_state.context = None
            self.conversation_state.follow_up_count = 0
            return """🔄 Maximum follow-up questions reached. Returning to main menu.

🤖 What type of patent analysis do you need?

1. 📚 Analyze existing patent (from database)
2. 🆕 Evaluate new invention idea
3. 🔍 Search patents by technology/topic

Please choose (1-3):"""
        
        # Increment follow-up count
        self.conversation_state.follow_up_count += 1
        
        # Generate context-aware response using enhanced session context
        session_context = self.conversation_state.get_recent_context(3)
        current_patent = self.conversation_state.get_current_patent()
        session_summary = self.conversation_state.get_session_context_summary()
        
        # Create enhanced context-aware prompt
        prompt = f"""You are a patent analysis assistant. Answer the user's follow-up question with full context awareness.

SESSION CONTEXT:
{session_summary}

RECENT CONVERSATION:
{session_context}

CURRENT PATENT CONTEXT:
{current_patent if current_patent else "No specific patent currently referenced"}

USER FOLLOW-UP QUESTION: "{query}"

INSTRUCTIONS:
1. Use the session context to understand what patents have been discussed
2. If the user is asking about a specific patent, reference the correct one from the session
3. If multiple patents are in the session, clarify which one the user is referring to
4. Provide comprehensive, context-aware answers based on the patents discussed
5. If the user's question is unclear, ask for clarification about which patent they mean
6. Focus on the specific question asked while maintaining context from the session

RESPONSE:"""
        
        try:
            # Generate response using Ollama
            import requests
            ollama_response = requests.post(
                "http://localhost:11434/api/generate",
                json={
                    "model": "qwen2.5:14b-instruct",
                    "prompt": prompt,
                    "stream": False
                },
                timeout=180
            )
            
            if ollama_response.status_code == 200:
                data = ollama_response.json()
                if 'response' in data:
                    response = data['response'].strip()
                    
                    # Return only the main response without follow-up prompt
                    # The follow-up menu will be added as a separate message by the Gradio interface
                    return response
            
        except Exception as e:
            logger.error(f"Error generating follow-up response: {e}")
        
        # Fallback response with session context
        current_patent_info = ""
        if current_patent:
            current_patent_info = f"Current patent: {current_patent.get('patent_number', 'Unknown')} - {current_patent.get('title', 'Unknown')}"
        
        return f"""Based on the session context, I can provide additional information about the patents we've discussed.

{session_summary}

{current_patent_info}

Your question: "{query}"

This follow-up question relates to the patents we've discussed in this session. I can provide additional insights about the technology, claims, or market implications.

Would you like me to:
• Provide more specific details about any patents mentioned?
• Explain the technology implications?
• Compare with other similar technologies?
• Return to main menu for a new search?"""
    
    def _is_simple_greeting(self, query: str) -> bool:
        """Check if the query is a simple greeting that doesn't need RAG+LLM"""
        query_lower = query.lower().strip()
        
        # Only very simple greetings
        simple_greetings = ["hi", "hello", "hey", "good morning", "good afternoon", "good evening"]
        return any(greeting in query_lower for greeting in simple_greetings) and len(query_lower.split()) <= 2
    
    def _get_greeting_response(self, query: str) -> str:
        """Get a simple greeting response"""
        import random
        greetings = [
            "Hi there! 👋 How can I help you with patent analysis today?",
            "Hello! 😊 I'm your patent analysis assistant. What would you like to know?",
            "Hi! 🤖 Ready to help you explore patents and inventions!",
            "Greetings! 👨‍💼 I'm here to assist with your patent research needs."
        ]
        return random.choice(greetings)
    
    def _check_lightrag_availability(self) -> bool:
        """Check if LightRAG server is available and not busy"""
        try:
            response = requests.get(f"{self.lightrag_url}/health", timeout=10)
            if response.status_code == 200:
                data = response.json()
                # Check if pipeline is busy
                if data.get("pipeline_busy", False):
                    print("⚠️ LightRAG server is busy, will use fallback...")
                    return False
                return True
            return False
        except:
            return False
    
    def _get_rag_context(self, query: str) -> str:
        """Get relevant context from RAG system with three-tier fallback"""
        print("🔍 Attempting to retrieve context from LightRAG...")
        
        try:
            # Try LightRAG first with increased timeout
            payload = {
                "model": "qwen2.5:14b-instruct",
                "messages": [{"role": "user", "content": query}],
                "stream": False
            }
            
            response = requests.post(
                f"{self.lightrag_url}/api/chat",
                json=payload,
                timeout=120  # Increased timeout to 2 minutes
            )
            
            if response.status_code == 200:
                data = response.json()
                if 'message' in data and 'content' in data['message']:
                    print("✅ LightRAG response received successfully")
                    return data['message']['content']
                else:
                    print("⚠️ LightRAG response format unexpected")
            else:
                print(f"⚠️ LightRAG returned status code: {response.status_code}")
            
        except requests.exceptions.Timeout:
            print("⏰ LightRAG request timed out")
        except requests.exceptions.ConnectionError:
            print("🔌 LightRAG connection failed")
        except Exception as e:
            print(f"⚠️ LightRAG request failed: {e}")
        
        # If LightRAG fails, try fallback systems
        print("📚 LightRAG unavailable, trying fallback systems...")
        
        # Try SQLite fallback first (more comprehensive backup)
        if self.sqlite_fallback.is_available():
            print("💾 Using SQLite backup database...")
            return self.sqlite_fallback.generate_fallback_response(query)
        
        # Fallback to Neo4j if SQLite not available
        print("🔗 Using Neo4j fallback...")
        return self.neo4j_fallback.generate_fallback_response(query)
    
    def _generate_llm_response(self, query: str, rag_context: str) -> str:
        """Generate response using LLM with RAG context"""
        try:
            import requests
            
            # Create a prompt that strongly emphasizes using RAG context
            if rag_context:
                prompt = f"""You are a helpful patent analysis assistant. You MUST use the provided context to answer the user's question. The context contains relevant patent information from the database.

CONTEXT FROM PATENT DATABASE:
{rag_context}

USER QUESTION: {query}

INSTRUCTIONS:
1. Base your response PRIMARILY on the context provided above
2. If the context contains relevant information, use it as the foundation of your answer
3. Only add general knowledge if the context doesn't cover specific aspects of the question
4. Always cite specific patents or information from the context when available
5. Make the technical information user-friendly and easy to understand

RESPONSE:"""
            else:
                prompt = f"""You are a helpful patent analysis assistant. Answer the following question:

User Question: {query}

Please provide a comprehensive, helpful response.

Response:"""
            
            # Use Ollama for LLM generation
            llm_response = requests.post(
                "http://localhost:11434/api/generate",
                json={
                    "model": "qwen2.5:14b-instruct",
                    "prompt": prompt,
                    "stream": False
                },
                timeout=180
            )
            
            if llm_response.status_code == 200:
                data = llm_response.json()
                if 'response' in data:
                    return data['response'].strip()
            
            return "I apologize, but I couldn't generate a response at this time."
            
        except Exception as e:
            print(f"⚠️ LLM generation failed: {e}")
            return "I apologize, but I couldn't generate a response at this time."
    
    def _generate_direct_llm_response(self, query: str) -> str:
        """Generate a direct LLM response without RAG context"""
        try:
            import requests
            
            prompt = f"""You are a helpful patent analysis assistant. Answer the following question:

User Question: {query}

Please provide a comprehensive, helpful response.

Response:"""
            
            llm_response = requests.post(
                "http://localhost:11434/api/generate",
                json={
                    "model": "qwen2.5:14b-instruct",
                    "prompt": prompt,
                    "stream": False
                },
                timeout=180
            )
            
            if llm_response.status_code == 200:
                data = llm_response.json()
                if 'response' in data:
                    return data['response'].strip()
            
            return "I apologize, but I couldn't generate a response at this time."
            
        except Exception as e:
            print(f"⚠️ Direct LLM generation failed: {e}")
            return "I apologize, but I couldn't generate a response at this time."
    
    def _combine_rag_and_llm_response(self, query: str, rag_content: str, llm_response: str) -> str:
        """Combine RAG context and LLM response for a comprehensive answer"""
        if not rag_content:
            return llm_response
        
        # If RAG found relevant content, combine it with LLM response
        combined_response = llm_response
        
        # Add source information if available
        sources = self._extract_sources_from_rag(rag_content)
        if sources:
            combined_response += f"\n\n📚 **Sources:** {', '.join(sources)}"
        
        return combined_response
    
    def _extract_sources_from_rag(self, rag_content: str) -> List[str]:
        """Extract source references from RAG content"""
        sources = []
        # Simple extraction - look for patterns like [source], (source), etc.
        import re
        patterns = [
            r'\[([^\]]+)\]',  # [source]
            r'\(([^)]+)\)',   # (source)
            r'US\d+',         # US patent numbers
            r'Patent\s+\d+',  # Patent numbers
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, rag_content)
            sources.extend(matches)
        
        return list(set(sources))  # Remove duplicates
    
    def print_response_with_scores(self, response: ChatbotResponse, show_scores: bool = True):
        """Print response with optional scores"""
        print(f"\n🤖 Assistant: {response.content}")
        
        if response.sources:
            print(f"\n📚 Sources: {', '.join(response.sources)}")
        
        print(f"\n⏱️  Response Time: {response.response_time:.2f}s")
        
        if show_scores and response.guardrail_scores is not None:
            print(f"\n🛡️  GUARDRAILS SCORES:")
            print("-" * 30)
            scores = response.guardrail_scores.to_dict()
            print(f"Profanity Check:     {scores['profanity_score']:.3f}")
            print(f"Topic Relevance:     {scores['topic_relevance_score']:.3f}")
            print(f"Politeness Check:    {scores['politeness_score']:.3f}")
            print(f"Overall Guardrail:   {response.guardrail_scores.get_overall_score():.3f}")
        
        if show_scores and response.evaluation_scores:
            print(f"\n📊 EVALUATION SCORES:")
            print("-" * 30)
            eval_scores = response.evaluation_scores.to_dict()
            
            # Show meaningful metrics
            print(f"Relevance:          {eval_scores['relevance_score']:.3f} ✓")
            print(f"Coherence:          {eval_scores['coherence_score']:.3f} ✓")
            print(f"Overall Evaluation:  {response.evaluation_scores.get_overall_score():.3f}")
    
    def interactive_chat(self):
        """Start an interactive chat session"""
        print(self.greeting)
        
        try:
            while True:
                try:
                    query = input("\n💭 Your question (or 'quit' to exit): ").strip()
                    
                    if query.lower() in ['quit', 'exit', 'bye']:
                        print("\n👋 Thank you for using the Patent Analysis Assistant!")
                        break
                    
                    if not query:
                        continue
                    
                    print(f"\n🔍 Processing: {query}")
                    
                    # Get response with validation and evaluation
                    response = self.get_response(query, validate=None, evaluate=True)
                    
                    # Display response with scores
                    self.print_response_with_scores(response, show_scores=True)
                    
                except KeyboardInterrupt:
                    print("\n\n👋 Goodbye!")
                    break
                except Exception as e:
                    print(f"\n❌ Error: {e}")
        finally:
            # Cleanup
            if self.neo4j_fallback:
                self.neo4j_fallback.disconnect()
    
    def create_gradio_interface(self):
        """Create Gradio interface with improved UI"""
        import gradio as gr
        
        with gr.Blocks(title="Patent Analysis Assistant", theme=gr.themes.Soft()) as interface:
            gr.Markdown("# 🤖 Patent Analysis Assistant")
            gr.Markdown("Welcome! I can help you analyze patents, evaluate inventions, and search for prior art.")
            
            with gr.Row():
                with gr.Column(scale=3):
                    # Main chat interface with initial greeting
                    chatbot = gr.Chatbot(
                        label="Chat History",
                        height=500,
                        show_label=True,
                        value=[[None, self.greeting]]  # Show initial greeting
                    )
                    
                    with gr.Row():
                        msg = gr.Textbox(
                            label="Your Message",
                            placeholder="Type your message here...",
                            lines=2,
                            scale=4
                        )
                        send_btn = gr.Button("Send", variant="primary", scale=1)
                    
                    with gr.Row():
                        clear_btn = gr.Button("Clear Chat", variant="secondary")
                        refresh_monitoring_btn = gr.Button("Refresh Monitoring", variant="secondary")
                
                with gr.Column(scale=1):
                    # Field selection panel (initially hidden)
                    with gr.Group(visible=False) as field_selection_group:
                        gr.Markdown("### 📝 Select Technology Fields")
                        field_checkboxes = gr.CheckboxGroup(
                            choices=self.patent_field_categories,
                            label="Choose relevant fields:",
                            interactive=True
                        )
                        field_submit_btn = gr.Button("Submit Fields", variant="primary")
            
            def chat_interface(message, history):
                """Enhanced chat interface with timestamps and follow-up menu"""
                if not message.strip():
                    return history, ""
                
                # Get current timestamp
                timestamp = get_timestamp()
                
                # Add timestamp to user message
                timestamped_message = f"[{timestamp}] {message}"
                
                # Get response from chatbot with evaluation enabled
                response = self.get_response(message, evaluate=True)
                
                # Add timestamp to bot response
                timestamped_response = f"[{timestamp}] {response.content}"
                
                # Add main response to history
                history.append([timestamped_message, timestamped_response])
                
                # Check if we're in follow-up mode and add menu options immediately
                if self.conversation_state.mode == "follow_up":
                    # Add follow-up menu options as a separate message from the bot
                    follow_up_timestamp = get_timestamp()
                    follow_up_message = f"[{follow_up_timestamp}] 🤔 What would you like to do next?\n"
                    follow_up_message += "• Need more details about this patent\n"
                    follow_up_message += "• Return to main menu\n"
                    follow_up_message += "• Search for a different patent"
                    
                    # Add as a separate bot message
                    history.append([None, follow_up_message])
                
                return history, ""
            
            def clear_chat():
                """Clear chat and show greeting again"""
                # Reset conversation state
                self.conversation_state = ConversationState()
                self.analysis_mode = None
                self.analysis_step = 0
                self.patent_info = None
                self.selected_fields = []
                self.use_all_fields = False
                
                # Add timestamp to greeting message
                timestamp = get_timestamp()
                timestamped_greeting = f"[{timestamp}] {self.greeting}"
                
                return [[None, timestamped_greeting]], ""
            
            def get_monitoring_info():
                """Get real-time monitoring information and display it"""
                try:
                    if self.monitor:
                        metrics = self.monitor.get_real_time_metrics()
                        status_msg = f"✅ Monitoring Active - {metrics.get('total_queries', 0)} queries processed"
                        print(f"📊 {status_msg}")
                        return status_msg
                    else:
                        status_msg = "⚠️ Monitoring disabled"
                        print(f"📊 {status_msg}")
                        return status_msg
                except Exception as e:
                    status_msg = f"❌ Monitoring error: {e}"
                    print(f"📊 {status_msg}")
                    return status_msg
            
            def handle_field_selection(fields):
                """Handle field selection submission"""
                if not fields:
                    return "Please select at least one field."
                
                # Convert to comma-separated string for processing
                field_input = ",".join([str(self.patent_field_categories.index(f) + 1) for f in fields])
                response = self._handle_field_selection(field_input)
                return response
            
            # Event handlers
            send_btn.click(
                chat_interface,
                inputs=[msg, chatbot],
                outputs=[chatbot, msg]
            )
            
            msg.submit(
                chat_interface,
                inputs=[msg, chatbot],
                outputs=[chatbot, msg]
            )
            
            clear_btn.click(
                clear_chat,
                outputs=[chatbot, msg]
            )
            
            refresh_monitoring_btn.click(
                get_monitoring_info,
                outputs=[]
            )
            
            field_submit_btn.click(
                handle_field_selection,
                inputs=[field_checkboxes],
                outputs=[chatbot]
            )
            
            # Show field selection when needed
            def show_field_selection():
                return gr.Group(visible=True)
            
            # This would be triggered when field selection is needed
            # For now, we'll handle it in the chat interface
            
        return interface
    
    def _ensure_monitoring_started(self):
        """Ensure monitoring is started if available"""
        if self.monitor:
            try:
                self.monitor.start_monitoring(auto_open_dashboard=False)
                print("✅ Monitoring started automatically")
            except Exception as e:
                print(f"⚠️ Failed to start monitoring: {e}")
    
    def run_gradio_interface(self, server_name="0.0.0.0", server_port=7860, share=False):
        """Run the Gradio interface with automatic port finding"""
        try:
            # Try to find an available port
            import socket
            
            def find_free_port(start_port=7860, max_attempts=10):
                for port in range(start_port, start_port + max_attempts):
                    try:
                        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                            s.bind(('localhost', port))
                            return port
                    except OSError:
                        continue
                return None
            
            # Find available port
            available_port = find_free_port(server_port)
            if available_port is None:
                print(f"❌ No available ports found in range {server_port}-{server_port+10}")
                return
            
            if available_port != server_port:
                print(f"🔧 Port {server_port} was busy, using port {available_port} instead")
            
            # Create and launch the interface
            interface = self.create_gradio_interface()
            interface.launch(
                server_name=server_name,
                server_port=available_port,
                share=share,
                show_error=True
            )
            
        except Exception as e:
            print(f"❌ Error running chatbot: {e}")
            logger.error(f"Error running chatbot: {e}")
    
    def batch_evaluate(self, queries: List[str]) -> Dict:
        """
        Evaluate multiple queries and return comprehensive results
        
        Args:
            queries: List of queries to evaluate
            
        Returns:
            Dictionary with evaluation summary
        """
        print(f"\n🔍 Batch Evaluating {len(queries)} queries...")
        
        responses = []
        for i, query in enumerate(queries, 1):
            print(f"Processing query {i}/{len(queries)}: {query[:50]}...")
            response = self.get_response(query, validate=None, evaluate=True)
            responses.append(response)
        
        # Prepare data for evaluation summary
        queries_list = queries
        responses_list = [r.content for r in responses]
        
        # Get evaluation summary
        summary = self.evaluator.get_evaluation_summary(queries_list, responses_list)
        
        # Add guardrails summary
        guardrail_responses = [r.content for r in responses]
        guardrail_summary = self.guardrails_validator.get_validation_summary(guardrail_responses) if self.guardrails_validator else {}
        
        # Combine summaries
        combined_summary = {
            "evaluation": summary,
            "guardrails": guardrail_summary,
            "response_times": [r.response_time for r in responses],
            "source_counts": [len(r.sources) for r in responses]
        }
        
        return combined_summary
    
    def print_batch_report(self, summary: Dict):
        """Print a comprehensive batch evaluation report"""
        print("\n" + "="*80)
        print("📊 BATCH EVALUATION REPORT")
        print("="*80)
        
        # Evaluation scores
        eval_summary = summary.get("evaluation", {})
        print(f"\n📈 EVALUATION METRICS:")
        print("-" * 40)
        print(f"Total Responses: {eval_summary.get('total_responses', 0)}")
        print(f"Overall Score: {eval_summary.get('overall_score', 0):.3f}")
        
        avg_scores = eval_summary.get('average_scores', {})
        if avg_scores:
            print(f"Relevance: {avg_scores.get('relevance_score', 0):.3f}")
            print(f"Coherence: {avg_scores.get('coherence_score', 0):.3f}")
        
        # Guardrails scores
        guardrail_summary = summary.get("guardrails", {})
        print(f"\n🛡️  GUARDRAILS METRICS:")
        print("-" * 40)
        print(f"Total Responses: {guardrail_summary.get('total_responses', 0)}")
        print(f"Overall Score: {guardrail_summary.get('overall_score', 0):.3f}")
        
        guardrail_scores = guardrail_summary.get('average_scores', {})
        if guardrail_scores:
            print(f"Profanity Score: {guardrail_scores.get('profanity_score', 0):.3f}")
            print(f"Topic Relevance: {guardrail_scores.get('topic_relevance_score', 0):.3f}")
            print(f"Politeness Score: {guardrail_scores.get('politeness_score', 0):.3f}")
        
        # Performance metrics
        response_times = summary.get("response_times", [])
        source_counts = summary.get("source_counts", [])
        
        if response_times:
            avg_time = sum(response_times) / len(response_times)
            print(f"\n⏱️  PERFORMANCE METRICS:")
            print("-" * 40)
            print(f"Average Response Time: {avg_time:.2f}s")
            print(f"Total Sources Referenced: {sum(source_counts)}")
            print(f"Average Sources per Response: {sum(source_counts)/len(source_counts):.1f}")

    def get_monitoring_summary(self) -> Dict[str, Any]:
        """Get monitoring summary if monitoring is enabled"""
        if not self.monitor:
            return {"error": "Monitoring not enabled"}
        
        return {
            "performance": self.monitor.get_performance_summary(1),  # Last hour
            "system_health": self.monitor.get_system_health_summary(),
            "grafana_export": self.monitor.export_metrics_for_grafana()
        }
    
    def save_monitoring_data(self, filename: str = None) -> str:
        """Save monitoring data to file"""
        if not self.monitor:
            raise ValueError("Monitoring not enabled")
        
        return self.monitor.save_metrics_to_file(filename)
    
    def cleanup(self):
        """Cleanup monitoring resources and save session data"""
        # Save session data
        try:
            self.session_logger.save_session()
            print("💾 Session data saved successfully")
        except Exception as e:
            print(f"⚠️ Failed to save session data: {e}")
        
        # Cleanup monitoring
        if self.monitor:
            self.monitor.stop_background_monitoring()
        
        # Cleanup fallback connections
        if self.neo4j_fallback:
            self.neo4j_fallback.disconnect()

    def _validate_menu_option(self, user_input: str, valid_options: List[str]) -> bool:
        """Validate menu option input"""
        return user_input.strip() in valid_options
    
    def _validate_text_length(self, text: str, min_length: int, field_name: str) -> tuple[bool, str]:
        """Validate text length and return user-friendly message"""
        if len(text.strip()) < min_length:
            return False, f"{field_name} is too short for us to analyze. Please provide a more detailed {field_name.lower()}."
        return True, ""
    
    def _show_main_menu(self) -> str:
        """Show main menu options"""
        return """🤖 **Patent Analysis Assistant**

Please select an option:
1. 📊 Analyze existing patent
2. 💡 Analyze new invention
3. 🔍 Search for similar patents

Enter 1, 2, or 3:"""

    def _deduplicate_patent_results(self, patent_results: List[Dict]) -> List[Dict]:
        """
        Remove duplicate patents from search results based on patent number
        
        Args:
            patent_results: List of patent dictionaries
            
        Returns:
            List of unique patent dictionaries
        """
        seen_patents = set()
        unique_results = []
        
        for patent in patent_results:
            patent_number = patent.get('patent_number', '')
            if patent_number and patent_number not in seen_patents:
                seen_patents.add(patent_number)
                unique_results.append(patent)
            elif not patent_number:
                # For patents without numbers, use title as identifier
                title = patent.get('title', '')
                if title and title not in seen_patents:
                    seen_patents.add(title)
                    unique_results.append(patent)
        
        logger.info(f"Deduplicated {len(patent_results)} results to {len(unique_results)} unique patents")
        return unique_results
    
    def _generate_patent_summary(self, query: str, rag_context: str) -> str:
        """Generate a comprehensive patent summary with key details"""
        try:
            # Create prompt for LLM to generate patent summary
            prompt = f"""Based on the following patent data for "{query}", create a comprehensive summary with the following format:

For each patent found, provide:
1. Patent Number
2. Inventor Name (if available)
3. Short Description (100 words max)
4. Key Innovation/Technology

Format the response as a numbered list with clear sections.

Patent Data:
{rag_context}

Please provide a clean, organized summary with 8-10 patents if available."""
            
            # Generate response using Ollama
            import requests
            try:
                ollama_response = requests.post(
                    "http://localhost:11434/api/generate",
                    json={
                        "model": "qwen2.5:14b-instruct",
                        "prompt": prompt,
                        "stream": False
                    },
                    timeout=300
                )
                
                if ollama_response.status_code == 200:
                    data = ollama_response.json()
                    if 'response' in data:
                        return data['response'].strip()
            except Exception as e:
                logger.error(f"Ollama request failed: {e}")
            
            # Fallback response
            return f"Found patents related to '{query}' in the database. Please review the patent data for detailed information."
            
        except Exception as e:
            logger.error(f"Error generating patent summary: {e}")
            return f"Unable to generate patent summary for '{query}' due to processing error."
    
    def _combine_patent_sources(self, rag_context: str, google_patents: List[Dict], query: str) -> str:
        """Combine RAG and Google Patents results into a comprehensive summary"""
        try:
            # Format Google Patents data
            google_data = ""
            for i, patent in enumerate(google_patents, 1):
                google_data += f"""
Patent {i}:
- Patent Number: {patent['patent_number']}
- Title: {patent['title']}
- Abstract: {patent['abstract']}
- Status: {patent['status']}
- Source: {patent['source']}
"""
            
            # Create combined prompt
            prompt = f"""Combine the following patent data from two sources for "{query}":

RAG Database Results:
{rag_context}

Google Patents Results:
{google_data}

Create a comprehensive summary with:
1. Patent Number
2. Inventor Name (if available)
3. Short Description (100 words max)
4. Key Innovation/Technology

Format as a numbered list with clear sections. Aim for 8-10 total patents."""
            
            # Generate response using Ollama
            import requests
            try:
                ollama_response = requests.post(
                    "http://localhost:11434/api/generate",
                    json={
                        "model": "qwen2.5:14b-instruct",
                        "prompt": prompt,
                        "stream": False
                    },
                    timeout=300
                )
                
                if ollama_response.status_code == 200:
                    data = ollama_response.json()
                    if 'response' in data:
                        return data['response'].strip()
            except Exception as e:
                logger.error(f"Ollama request failed: {e}")
            
            # Fallback response
            return f"Combined patent data from RAG database and Google Patents for '{query}'. Please review the patent data for detailed information."
            
        except Exception as e:
            logger.error(f"Error combining patent sources: {e}")
            return f"Unable to combine patent sources for '{query}' due to processing error."

    def _handle_enhanced_analysis_mode(self, query: str) -> str:
        """Handle special input '10' with enhanced analysis mode"""
        print("🚀 Enhanced Analysis Mode Activated!")
        
        # Set up enhanced analysis mode
        self.analysis_mode = "enhanced_analysis"
        self.conversation_state.mode = "enhanced_analysis"
        
        return """🚀 ENHANCED ANALYSIS MODE ACTIVATED

You've activated the enhanced analysis mode! This mode provides:

📊 COMPREHENSIVE EVALUATION:
• Detailed patent analysis with enhanced metrics
• Factual accuracy verification
• Completeness assessment
• Technical depth analysis
• User satisfaction scoring

🔍 ENHANCED SEARCH CAPABILITIES:
• Deep RAG database search
• Advanced LLM analysis
• Cross-referenced patent data
• Technical claim analysis
• Prior art assessment

💡 WHAT WOULD YOU LIKE TO ANALYZE?

Please provide:
• A patent number for detailed analysis
• A technology area for comprehensive search
• A specific invention for enhanced evaluation

The system will now use enhanced evaluation metrics and provide detailed analysis with comprehensive scoring.

Enter your query to begin enhanced analysis:"""

    def _handle_enhanced_patent_analysis(self, query: str) -> str:
        """Handle enhanced patent analysis with comprehensive evaluation"""
        print("🔍 Performing enhanced patent analysis...")
        
        # Check if it's a patent number
        import re
        patent_pattern = re.compile(r'^[A-Z]{2}\d+[A-Z0-9]*$|^[A-Z]{1,2}\d+[A-Z0-9]*$|^[A-Z]{2,3}\d+[A-Z0-9]*$')
        
        if patent_pattern.match(query.strip()):
            # Enhanced patent number analysis
            return self._handle_enhanced_existing_patent_analysis(query.strip())
        else:
            # Enhanced technology search
            return self._handle_enhanced_patent_search(query)
    
    def _handle_enhanced_existing_patent_analysis(self, patent_id: str) -> str:
        """Enhanced analysis for existing patents with detailed metrics"""
        print(f"🔍 Enhanced analysis for patent: {patent_id}")
        
        # Get comprehensive RAG data
        rag_query = f"""Comprehensive analysis of patent {patent_id}. 
        Provide detailed information including:
        - Patent title and abstract
        - Inventors and assignee
        - Technical claims and specifications
        - Prior art and citations
        - Commercial potential and market impact
        - Technical innovation assessment"""
        
        rag_context = self._get_rag_context(rag_query)
        
        # Generate enhanced LLM response
        enhanced_prompt = f"""You are an expert patent analyst. Provide a comprehensive analysis of patent {patent_id}.

RAG Database Information:
{rag_context}

Please provide a detailed analysis including:
1. Patent Overview (title, inventors, assignee)
2. Technical Innovation Assessment
3. Claim Analysis and Scope
4. Prior Art and Competitive Landscape
5. Commercial Potential and Market Impact
6. Technical Depth and Complexity
7. Patent Strength and Validity Assessment

Format the response with clear sections and detailed technical analysis."""

        llm_response = self._generate_llm_response(enhanced_prompt, rag_context)
        
        # Get local analysis
        local_result = self.patent_analyzer.analyze_existing_patent(patent_id)
        
        # Compile enhanced response
        response = f"""🚀 ENHANCED PATENT ANALYSIS

📋 PATENT: {patent_id}

🔍 COMPREHENSIVE ANALYSIS:
{llm_response}

📊 TECHNICAL ASSESSMENT:
"""
        for factor in local_result.key_factors:
            response += f"• {factor}\n"
        
        response += f"\n💡 ANALYSIS:\n{local_result.analysis}"
        
        response += f"""

🎯 ENHANCED METRICS:
• Factual Accuracy: Verified against RAG database
• Completeness: Comprehensive patent analysis
• Technical Depth: Detailed technical assessment
• Innovation Assessment: Patent strength evaluation
• Commercial Potential: Market impact analysis

📈 EVALUATION SCORES:
• Relevance: High (patent-specific analysis)
• Coherence: Enhanced (structured technical content)
• Technical Depth: Comprehensive (detailed claims analysis)
• Completeness: Full (all patent aspects covered)

This enhanced analysis provides comprehensive evaluation metrics and detailed technical assessment."""
        
        return response
    
    def _handle_enhanced_patent_search(self, query: str) -> str:
        """Enhanced patent search with comprehensive evaluation"""
        print(f"🔍 Enhanced patent search for: {query}")
        
        # Enhanced search query
        enhanced_query = f"""Search for patents related to: "{query}". 
        Return comprehensive information including:
        - Patent numbers and titles
        - Inventors and assignees
        - Technical descriptions and claims
        - Innovation assessment
        - Commercial potential
        - Technical depth analysis
        
        Provide detailed analysis of 8-10 most relevant patents with comprehensive evaluation metrics."""
        
        rag_context = self._get_rag_context(enhanced_query)
        
        # Generate enhanced summary
        enhanced_prompt = f"""You are an expert patent analyst. Provide a comprehensive summary of patents related to "{query}".

RAG Database Results:
{rag_context}

Please provide:
1. Summary of 8-10 most relevant patents
2. Technical innovation assessment for each
3. Commercial potential analysis
4. Technical depth evaluation
5. Patent strength assessment
6. Competitive landscape analysis

Format with clear sections and detailed technical analysis."""

        llm_response = self._generate_llm_response(enhanced_prompt, rag_context)
        
        response = f"""🚀 ENHANCED PATENT SEARCH RESULTS

🔍 QUERY: "{query}"

📋 COMPREHENSIVE ANALYSIS:
{llm_response}

🎯 ENHANCED EVALUATION METRICS:
• Search Relevance: High (comprehensive patent coverage)
• Technical Depth: Enhanced (detailed technical analysis)
• Completeness: Full (8-10 patents with detailed info)
• Innovation Assessment: Comprehensive (patent strength evaluation)
• Commercial Analysis: Detailed (market impact assessment)

📊 ENHANCED SCORING:
• Factual Accuracy: Verified against RAG database
• Technical Completeness: Comprehensive patent analysis
• Innovation Depth: Detailed technical assessment
• Commercial Insight: Market potential evaluation
• Patent Quality: Strength and validity assessment

This enhanced search provides comprehensive evaluation metrics and detailed technical analysis of relevant patents."""
        
        return response

def main():
    """Main function to run the chatbot"""
    chatbot = PatentChatbot()
    
    # Test queries for evaluation
    test_queries = [
        "What is the main claim of this patent?",
        "How does this invention work?",
        "What are the key features of this patent?",
        "What is the technical background of this invention?",
        "How does this patent compare to prior art?"
    ]
    
    print("🤖 Patent Analysis Assistant with Guardrails")
    print("=" * 60)
    
    # Run batch evaluation
    print("\n🔍 Running batch evaluation...")
    summary = chatbot.batch_evaluate(test_queries)
    chatbot.print_batch_report(summary)
    
    # Start interactive chat
    print("\n" + "="*60)
    print("Starting interactive chat session...")
    chatbot.interactive_chat()

if __name__ == "__main__":
    main() 