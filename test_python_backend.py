#!/usr/bin/env python3
import requests
import json
import time

def test_backend():
    base_url = "http://localhost:5000"
    
    print("🧪 Testing Python Backend...")
    
    # Test health endpoint
    try:
        response = requests.get(f"{base_url}/health", timeout=5)
        print(f"✅ Health check: {response.status_code} - {response.json()}")
    except Exception as e:
        print(f"❌ Health check failed: {e}")
        return False
    
    # Test analysis endpoint
    test_data = {
        "content": "The chairman announced that his company will hire more diverse candidates. Women and minorities are encouraged to apply.",
        "analysisType": "comprehensive",
        "sensitivity": "standard",
        "inputType": "text"
    }
    
    try:
        response = requests.post(
            f"{base_url}/api/analyze",
            headers={"Content-Type": "application/json"},
            json=test_data,
            timeout=30
        )
        print(f"✅ Analysis endpoint: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"📊 Overall Score: {result.get('overallScore', 'N/A')}")
            print(f"📊 Gender Score: {result.get('genderScore', 'N/A')}")
            print(f"📊 Analysis ID: {result.get('id', 'N/A')}")
            return True
        else:
            print(f"❌ Analysis failed: {response.text}")
            return False
    except Exception as e:
        print(f"❌ Analysis request failed: {e}")
        return False

if __name__ == "__main__":
    # Wait a moment for server to start
    time.sleep(2)
    test_backend()