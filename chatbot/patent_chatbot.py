#!/usr/bin/env python3
"""
Patent Analysis Chatbot Interface
Interactive chatbot for querying G06 patents using LightRAG
"""

import gradio as gr
import requests
import json
import logging
import re
from typing import Iterator, Dict, Any
import time

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuration
LIGHTRAG_BASE_URL = "http://localhost:9621"
CHAT_MODEL = "qwen2.5:14b-instruct"

class PatentChatbot:
    def __init__(self):
        self.conversation_history = []
        self.base_url = LIGHTRAG_BASE_URL
    
    def check_lightrag_health(self) -> bool:
        """Check if LightRAG server is healthy"""
        try:
            response = requests.get(f"{self.base_url}/health", timeout=10)
            if response.status_code == 200:
                data = response.json()
                return data.get("status") == "healthy"
            return False
        except Exception as e:
            logger.error(f"Error checking LightRAG health: {e}")
            return False
    
    def get_document_count(self) -> int:
        """Get the number of documents in LightRAG"""
        try:
            response = requests.get(f"{self.base_url}/documents", timeout=10)
            if response.status_code == 200:
                data = response.json()
                processed = len(data.get("statuses", {}).get("processed", []))
                processing = len(data.get("statuses", {}).get("processing", []))
                return processed + processing
            return 0
        except Exception as e:
            logger.error(f"Error getting document count: {e}")
            return 0
    
    def chat_with_lightrag(self, message: str, history: list) -> Iterator[str]:
        """Send a message to LightRAG and stream the response"""
        if not message.strip():
            yield "Please enter a message."
            return
        
        # Check if LightRAG is available
        if not self.check_lightrag_health():
            yield "❌ LightRAG server is not available. Please ensure the server is running."
            return
        
        try:
            # Prepare the chat request
            chat_request = {
                "model": CHAT_MODEL,
                "messages": [
                    {
                        "role": "user",
                        "content": message
                    }
                ],
                "stream": True
            }
            
            # Send request to LightRAG
            response = requests.post(
                f"{self.base_url}/api/chat",
                json=chat_request,
                timeout=900,  # 15 minutes timeout for heavy model
                stream=True
            )
            
            # If streaming fails, try non-streaming
            if response.status_code != 200:
                logger.warning("Streaming failed, trying non-streaming request")
                chat_request["stream"] = False
                response = requests.post(
                    f"{self.base_url}/api/chat",
                    json=chat_request,
                    timeout=900
                )
            
            if response.status_code == 200:
                # Check if response is streaming or non-streaming
                if chat_request.get("stream", True):
                    # Stream the response
                    full_response = ""
                    for line in response.iter_lines():
                        if line:
                            try:
                                # Parse direct JSON format (LightRAG style)
                                chunk = json.loads(line.decode('utf-8'))
                                
                                # Handle LightRAG format
                                if 'message' in chunk and 'content' in chunk['message']:
                                    content = chunk['message']['content']
                                    full_response += content
                                    yield full_response
                                # Handle standard OpenAI format
                                elif 'choices' in chunk and chunk['choices']:
                                    delta = chunk['choices'][0].get('delta', {})
                                    if 'content' in delta:
                                        content = delta['content']
                                        full_response += content
                                        yield full_response
                                # Check for completion
                                elif chunk.get('done', False):
                                    break
                            except json.JSONDecodeError:
                                # Try SSE format as fallback
                                if line.startswith(b'data: '):
                                    data = line[6:].decode('utf-8')
                                    if data.strip() == '[DONE]':
                                        break
                                    try:
                                        chunk = json.loads(data)
                                        if 'choices' in chunk and chunk['choices']:
                                            delta = chunk['choices'][0].get('delta', {})
                                            if 'content' in delta:
                                                content = delta['content']
                                                full_response += content
                                                yield full_response
                                    except json.JSONDecodeError:
                                        continue
                            except Exception as e:
                                logger.error(f"Error parsing response chunk: {e}")
                                continue
                    
                    # Post-process the response to consolidate source references
                    if full_response.strip():
                        processed_response = self._consolidate_source_references(full_response)
                        yield processed_response
                    else:
                        yield "No response received from LightRAG. Please try a different query."
                else:
                    # Handle non-streaming response
                    try:
                        data = response.json()
                        if 'message' in data and 'content' in data['message']:
                            content = data['message']['content']
                            yield content
                        else:
                            yield "No response content found in LightRAG response."
                    except Exception as e:
                        logger.error(f"Error parsing non-streaming response: {e}")
                        yield "Error parsing LightRAG response."
            else:
                error_msg = f"❌ Error: {response.status_code} - {response.text}"
                yield error_msg
                
        except requests.exceptions.Timeout:
            yield "⏰ Request timed out. Please try a shorter or more specific query."
        except requests.exceptions.ConnectionError:
            yield "❌ Cannot connect to LightRAG server. Please ensure it's running."
        except Exception as e:
            logger.error(f"Error in chat: {e}")
            yield f"❌ Error: {str(e)}"
    
    def _consolidate_source_references(self, response: str) -> str:
        """Consolidate multiple references to the same source into a single reference"""
        
        # Define source patterns to look for
        source_patterns = [
            r'\[KG\]\s*Harvard USPTO Dataset',
            r'\[KG\]\s*HUPD',
            r'\[KG\]\s*https://patentdataset\.org/',
            r'Source:\s*Harvard USPTO Dataset',
            r'Reference:\s*https://patentdataset\.org/'
        ]
        
        # Check if any source references exist
        has_source_references = any(re.search(pattern, response, re.IGNORECASE) for pattern in source_patterns)
        
        if not has_source_references:
            return response
        
        # Remove all existing source references from the response
        cleaned_response = response
        for pattern in source_patterns:
            cleaned_response = re.sub(pattern, '', cleaned_response, flags=re.IGNORECASE)
        
        # Clean up extra whitespace and newlines
        cleaned_response = re.sub(r'\n\s*\n\s*\n', '\n\n', cleaned_response)  # Remove extra blank lines
        cleaned_response = re.sub(r'^\s+', '', cleaned_response, flags=re.MULTILINE)  # Remove leading whitespace
        cleaned_response = cleaned_response.strip()
        
        # Add a single consolidated reference at the end
        consolidated_reference = "\n\n---\n**Source:** Harvard USPTO Dataset (HUPD) - https://patentdataset.org/\n**Citation:** Suzgun, M., et al. (2022). The Harvard USPTO Patent Dataset: A Large-Scale, Well-Structured, and Multi-Purpose Corpus of Patent Applications. arXiv preprint arXiv:2207.04043"
        
        return cleaned_response + consolidated_reference
    
    def get_system_status(self) -> str:
        """Get system status information"""
        status_lines = []
        
        # Check LightRAG health
        if self.check_lightrag_health():
            status_lines.append("✅ LightRAG Server: Running")
        else:
            status_lines.append("❌ LightRAG Server: Not available")
        
        # Get document count
        doc_count = self.get_document_count()
        status_lines.append(f"📄 Documents in LightRAG: {doc_count}")
        
        # Check Ollama
        try:
            ollama_response = requests.get("http://localhost:11434/api/tags", timeout=5)
            if ollama_response.status_code == 200:
                models = ollama_response.json().get('models', [])
                qwen_model = next((m for m in models if m.get('name') == CHAT_MODEL), None)
                if qwen_model:
                    status_lines.append(f"✅ Ollama Model: {CHAT_MODEL}")
                else:
                    status_lines.append(f"❌ Ollama Model: {CHAT_MODEL} not found")
            else:
                status_lines.append("❌ Ollama: Not responding")
        except:
            status_lines.append("❌ Ollama: Not available")
        
        return "\n".join(status_lines)
    
    def create_interface(self):
        """Create the Gradio interface"""
        with gr.Blocks(
            title="Patent Analysis Chatbot",
            theme=gr.themes.Soft(),
            css="""
            .gradio-container {
                max-width: 1200px !important;
                margin: auto !important;
            }
            .chat-container {
                height: 600px;
            }
            """
        ) as interface:
            
            gr.Markdown("""
            # 🔬 Patent Analysis Chatbot
            
            **Query G06 patents using LightRAG and AI-powered analysis**
            
            This chatbot allows you to ask questions about computer technology patents (G06 classification) 
            and get intelligent responses based on the Harvard USPTO Dataset (HUPD).
            
            **Data Source:** Harvard USPTO Dataset (HUPD) - https://patentdataset.org/
            **Citation:** Suzgun, M., et al. (2022). The Harvard USPTO Patent Dataset: A Large-Scale, Well-Structured, and Multi-Purpose Corpus of Patent Applications. arXiv preprint arXiv:2207.04043
            **License:** CreativeCommons Attribution-NonCommercial-ShareAlike 4.0 International
            """)
            
            with gr.Row():
                with gr.Column(scale=3):
                    # Chat interface
                    chatbot = gr.Chatbot(
                        label="Patent Analysis Chat",
                        height=500,
                        show_label=True,
                        container=True,
                        bubble_full_width=False
                    )
                    
                    with gr.Row():
                        msg = gr.Textbox(
                            label="Ask about G06 patents...",
                            placeholder="e.g., What are recent patents in computer vision?",
                            scale=4
                        )
                        send_btn = gr.Button("Send", variant="primary", scale=1)
                    
                    # Clear button
                    clear_btn = gr.Button("Clear Chat", variant="secondary")
                
                with gr.Column(scale=1):
                    # Status panel
                    gr.Markdown("### 📊 System Status")
                    status_display = gr.Textbox(
                        label="Status",
                        value="Checking system status...",
                        interactive=False,
                        lines=8
                    )
                    refresh_status_btn = gr.Button("🔄 Refresh Status", size="sm")
                    
                    # Example queries
                    gr.Markdown("### 💡 Example Queries")
                    example_queries = [
                        "What are the latest patents in machine learning?",
                        "Show me patents related to computer vision technology",
                        "What innovations exist in natural language processing?",
                        "Find patents about blockchain technology",
                        "What are the trends in cybersecurity patents?"
                    ]
                    
                    for query in example_queries:
                        gr.Markdown(f"• `{query}`")
            
            # Event handlers
            def user_input(message, history):
                return "", history + [[message, None]]
            
            def bot_response(history):
                if history and history[-1][1] is None:
                    message = history[-1][0]
                    history[-1][1] = ""
                    for response in self.chat_with_lightrag(message, history):
                        history[-1][1] = response
                        yield history
            
            def clear_chat():
                return []
            
            def update_status():
                return self.get_system_status()
            
            # Connect events
            msg.submit(
                user_input,
                [msg, chatbot],
                [msg, chatbot],
                queue=False
            ).then(
                bot_response,
                chatbot,
                chatbot
            )
            
            send_btn.click(
                user_input,
                [msg, chatbot],
                [msg, chatbot],
                queue=False
            ).then(
                bot_response,
                chatbot,
                chatbot
            )
            
            clear_btn.click(clear_chat, outputs=chatbot)
            refresh_status_btn.click(update_status, outputs=status_display)
            
            # Initial status update
            interface.load(update_status, outputs=status_display)
        
        return interface

def main():
    """Main function to run the chatbot"""
    chatbot = PatentChatbot()
    interface = chatbot.create_interface()
    
    # Launch the interface
    interface.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
        quiet=False
    )

if __name__ == "__main__":
    main() 