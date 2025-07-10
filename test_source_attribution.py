#!/usr/bin/env python3
"""
Test script to verify source attribution is working properly
"""

import requests
import json

def test_source_attribution():
    """Test if LightRAG responses include proper source attribution"""
    
    print("🔍 Testing source attribution in LightRAG responses...")
    
    # Test query
    test_query = {
        "model": "qwen2.5:14b-instruct",
        "messages": [{"role": "user", "content": "What are some recent patents in computer vision technology?"}],
        "stream": False
    }
    
    try:
        response = requests.post(
            "http://localhost:9621/api/chat",
            json=test_query,
            timeout=60
        )
        
        if response.status_code == 200:
            data = response.json()
            if 'message' in data and 'content' in data['message']:
                content = data['message']['content']
                print("\n📝 Response:")
                print(content)
                
                # Check for source references
                if "[KG]" in content:
                    print("\n✅ Found Knowledge Graph references in response")
                else:
                    print("\n⚠️ No Knowledge Graph references found")
                
                # Check for Harvard USPTO references
                if "Harvard USPTO" in content or "HUPD" in content:
                    print("✅ Found Harvard USPTO Dataset references")
                else:
                    print("⚠️ No Harvard USPTO Dataset references found")
                    
            else:
                print("❌ No content found in response")
        else:
            print(f"❌ Error: {response.status_code} - {response.text}")
            
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    test_source_attribution() 