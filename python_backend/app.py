from flask import Flask, request, jsonify
from flask_cors import CORS
import os
from dotenv import load_dotenv
from bias_detector import BiasDetector
from url_scraper import URLScraper
import json
from datetime import datetime

load_dotenv()

app = Flask(__name__)
CORS(app)

bias_detector = BiasDetector()
url_scraper = URLScraper()

analyses_storage = []
analysis_id_counter = 1

@app.route('/api/analyze', methods=['POST'])
def analyze_content():
    global analysis_id_counter
    
    try:
        data = request.json
        content = data.get('content', '')
        url = data.get('url', '')
        analysis_type = data.get('analysisType', 'comprehensive')
        sensitivity = data.get('sensitivity', 'standard')
        input_type = data.get('inputType', 'text')
        
        if input_type == 'url' and url:
            try:
                scraped_data = url_scraper.scrape_article(url)
                content = scraped_data['content']
                word_count = scraped_data['word_count']
                title = scraped_data.get('title', 'Untitled')
            except Exception as e:
                return jsonify({'error': f'Failed to scrape URL: {str(e)}'}), 400
        else:
            word_count = len(content.split())
            title = None
        
        # Perform bias analysis
        analysis_result = bias_detector.analyze_bias(
            content=content,
            analysis_type=analysis_type,
            sensitivity=sensitivity
        )
        
        # Create analysis record
        analysis_record = {
            'id': analysis_id_counter,
            'content': content[:500] + '...' if len(content) > 500 else content,
            'url': url if input_type == 'url' else None,
            'analysisType': analysis_type,
            'sensitivity': sensitivity,
            'inputType': input_type,
            'overallScore': analysis_result['overall_score'],
            'genderScore': analysis_result['gender_score'],
            'racialScore': analysis_result['racial_score'],
            'politicalScore': analysis_result['political_score'],
            'culturalScore': analysis_result['cultural_score'],
            'detailedReport': analysis_result['detailed_report'],
            'wordCount': word_count,
            'createdAt': datetime.now().isoformat(),
            'title': title
        }
        
        analyses_storage.append(analysis_record)
        analysis_id_counter += 1
        
        return jsonify(analysis_record)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/analyses', methods=['GET'])
def get_analyses():
    try:
        limit = int(request.args.get('limit', 10))
        # Sort by creation date (newest first) and limit results
        sorted_analyses = sorted(analyses_storage, key=lambda x: x['createdAt'], reverse=True)
        return jsonify(sorted_analyses[:limit])
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/analyses/<int:analysis_id>', methods=['GET'])
def get_analysis(analysis_id):
    try:
        analysis = next((a for a in analyses_storage if a['id'] == analysis_id), None)
        if analysis:
            return jsonify(analysis)
        else:
            return jsonify({'error': 'Analysis not found'}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy', 'service': 'BiasGuard Python Backend'})

if __name__ == '__main__':
    print("🚀 Starting BiasGuard Python Backend...")
    print("🔧 Loading AI models...")
    
    # Initialize the bias detector (this may take some time for model loading)
    bias_detector.initialize()
    
    print("✅ Backend ready!")
    app.run(host='0.0.0.0', port=5000, debug=True)