#!/usr/bin/env python3

import os
import sys
from dotenv import load_dotenv

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

load_dotenv()

from app import app

if __name__ == '__main__':
    print("🚀 Starting Bias Buster Python Backend...")
    print("🔧 Environment: Development")
    print("🌐 Server: http://localhost:5000")
    print("📡 CORS: Enabled for frontend")
    print("-" * 50)
    
    groq_key = os.getenv('GROQ_API_KEY')
    if groq_key and groq_key != 'your_groq_api_key_here':
        print("✅ GROQ API Key loaded")
    else:
        print("⚠️  GROQ API Key not configured - using fallback analysis")
    
    app.run(host='0.0.0.0', port=5000, debug=True)