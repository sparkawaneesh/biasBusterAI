import os
import json
import re
from typing import Dict, List, Any
try:
    from groq import Groq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False
    print("⚠️ GROQ not available, using fallback analysis")

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    print("⚠️ NumPy not available, using basic analysis")

class BiasDetector:
    def __init__(self):
        self.groq_api_key = os.getenv('GROQ_API_KEY') or os.getenv('GROQ_API_KEY_ENV_VAR')
        self.groq_client = None
        self.local_model = None
        self.tokenizer = None
        self.sentiment_pipeline = None
        self.bias_keywords = self._load_bias_keywords()
        print(f"🔑 GROQ API Key status: {'✅ Loaded' if self.groq_api_key and len(self.groq_api_key) > 10 else '❌ Missing'}")
        
    def initialize(self):
        print("🔧 Initializing GROQ client...")
        if self.groq_api_key and GROQ_AVAILABLE:
            try:
                self.groq_client = Groq(api_key=self.groq_api_key)
                print("✅ GROQ client initialized")
            except Exception as e:
                print(f"⚠️ GROQ initialization failed: {e}")
                self.groq_client = None
        else:
            print("⚠️ GROQ API key not found or GROQ not available, using local analysis only")
        
        print("✅ Bias detector ready with keyword-based analysis")
            
    def _load_bias_keywords(self) -> Dict[str, List[str]]:
        return {
            'gender': [
                'he', 'she', 'his', 'her', 'him', 'man', 'woman', 'male', 'female',
                'masculine', 'feminine', 'boy', 'girl', 'gentleman', 'lady',
                'chairman', 'chairwoman', 'spokesman', 'spokeswoman', 'policeman', 'policewoman'
            ],
            'racial': [
                'race', 'ethnic', 'minority', 'majority', 'white', 'black', 'asian', 'hispanic',
                'latino', 'african', 'european', 'american', 'native', 'indigenous',
                'immigrant', 'foreigner', 'diversity', 'multicultural'
            ],
            'political': [
                'liberal', 'conservative', 'democrat', 'republican', 'left', 'right',
                'progressive', 'traditional', 'socialism', 'capitalism', 'government',
                'policy', 'election', 'vote', 'political', 'partisan'
            ],
            'cultural': [
                'culture', 'religion', 'christian', 'muslim', 'jewish', 'hindu', 'buddhist',
                'secular', 'spiritual', 'belief', 'tradition', 'customs', 'heritage',
                'community', 'society', 'values', 'lifestyle'
            ]
        }
    
    def analyze_bias(self, content: str, analysis_type: str = 'comprehensive', 
                    sensitivity: str = 'standard') -> Dict[str, Any]:
        if self.groq_client:
            return self._analyze_with_groq(content, analysis_type, sensitivity)
        else:
            return self._analyze_locally(content, analysis_type, sensitivity)
    
    def _analyze_with_groq(self, content: str, analysis_type: str, sensitivity: str) -> Dict[str, Any]:
        try:
            prompt = self._build_groq_prompt(content, analysis_type, sensitivity)
            
            response = self.groq_client.chat.completions.create(
                model="llama3-8b-8192",
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert bias detection AI that analyzes text for various forms of bias including gender, racial, political, and cultural bias. Provide detailed, actionable feedback in JSON format."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.1,
                max_tokens=2000,
            )
            
            response_content = response.choices[0].message.content
            return self._parse_groq_response(response_content, content)
            
        except Exception as e:
            print(f"GROQ API error: {e}")
            return self._analyze_locally(content, analysis_type, sensitivity)
    
    def _build_groq_prompt(self, content: str, analysis_type: str, sensitivity: str) -> str:
        focus_area = "all types of bias (gender, racial, political, cultural)" if analysis_type == 'comprehensive' else f"{analysis_type} bias specifically"
        
        return f"""
Analyze the following text for {focus_area} with {sensitivity} sensitivity level.

Text to analyze:
"{content}"

Please provide your analysis in the following JSON format:
{{
  "overall_score": [0-10 score],
  "gender_score": [0-10 score],
  "racial_score": [0-10 score], 
  "political_score": [0-10 score],
  "cultural_score": [0-10 score],
  "sections": [
    {{
      "type": "gender|racial|political|cultural",
      "riskLevel": "low|medium|high",
      "issues": [
        {{
          "description": "Description of the issue",
          "location": "Specific text location or line reference",
          "severity": "low|medium|high"
        }}
      ],
      "recommendations": [
        "Specific recommendation for improvement"
      ],
      "suggestedRevision": "Improved version of problematic text"
    }}
  ],
  "summary": {{
    "priorityActions": ["Action 1", "Action 2"],
    "strengths": ["Strength 1", "Strength 2"],
    "impact": "Description of expected impact from improvements"
  }}
}}

Scoring guidelines:
- 0-3: Low bias/risk
- 4-6: Medium bias/risk  
- 7-10: High bias/risk

Focus on constructive feedback and actionable recommendations.
        """
    
    def _parse_groq_response(self, response_content: str, original_content: str) -> Dict[str, Any]:
        """Parse GROQ API response"""
        try:
            # Extract JSON from response
            json_match = re.search(r'\{.*\}', response_content, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())
                return {
                    'overall_score': result.get('overall_score', 0),
                    'gender_score': result.get('gender_score', 0),
                    'racial_score': result.get('racial_score', 0),
                    'political_score': result.get('political_score', 0),
                    'cultural_score': result.get('cultural_score', 0),
                    'detailed_report': result
                }
        except Exception as e:
            print(f"Error parsing GROQ response: {e}")
        
        # Fallback response
        return self._generate_fallback_response(original_content)
    
    def _analyze_locally(self, content: str, analysis_type: str, sensitivity: str) -> Dict[str, Any]:
        """Analyze bias using local models and keyword detection"""
        
        # Tokenize and preprocess content
        words = content.lower().split()
        word_count = len(words)
        
        # Calculate bias scores for each category
        scores = {}
        sections = []
        
        for bias_type, keywords in self.bias_keywords.items():
            if analysis_type != 'comprehensive' and analysis_type != bias_type:
                scores[f'{bias_type}_score'] = 0
                continue
                
            # Count keyword matches
            matches = [word for word in words if any(keyword in word for keyword in keywords)]
            match_ratio = len(matches) / max(word_count, 1)
            
            # Calculate bias score based on sensitivity
            base_score = min(match_ratio * 100, 10)
            
            if sensitivity == 'high':
                bias_score = min(base_score * 1.5, 10)
            elif sensitivity == 'low':
                bias_score = base_score * 0.7
            else:  # standard
                bias_score = base_score
            
            scores[f'{bias_type}_score'] = round(bias_score, 1)
            
            # Generate sections for detailed report
            if bias_score > 2:
                risk_level = 'high' if bias_score > 7 else 'medium' if bias_score > 4 else 'low'
                sections.append({
                    'type': bias_type,
                    'riskLevel': risk_level,
                    'issues': [{
                        'description': f'Detected {bias_type} bias indicators in the text',
                        'location': f'Keywords found: {", ".join(matches[:5])}',
                        'severity': risk_level
                    }],
                    'recommendations': [
                        f'Review language for {bias_type} bias patterns',
                        'Consider using more inclusive terminology',
                        'Ensure balanced representation'
                    ],
                    'suggestedRevision': f'Consider rephrasing sections that contain {bias_type} bias indicators'
                })
        
        # Calculate overall score
        if NUMPY_AVAILABLE:
            overall_score = np.mean(list(scores.values()))
        else:
            overall_score = sum(scores.values()) / len(scores.values())
        scores['overall_score'] = round(overall_score, 1)
        
        # Generate summary
        summary = {
            'priorityActions': [
                'Review flagged content sections',
                'Apply suggested language improvements',
                'Consider diverse perspectives'
            ],
            'strengths': [
                'Content structure is clear',
                'Analysis completed successfully'
            ],
            'impact': 'Implementing these changes will improve content inclusivity and reduce bias'
        }
        
        detailed_report = {
            'sections': sections,
            'summary': summary,
            'analysis_method': 'local_keyword_detection',
            'sensitivity': sensitivity
        }
        
        return {
            **scores,
            'detailed_report': detailed_report
        }
    
    def _generate_fallback_response(self, content: str) -> Dict[str, Any]:
        """Generate fallback response when other methods fail"""
        return {
            'overall_score': 5.0,
            'gender_score': 5.0,
            'racial_score': 5.0,
            'political_score': 5.0,
            'cultural_score': 5.0,
            'detailed_report': {
                'sections': [],
                'summary': {
                    'priorityActions': ['Unable to complete detailed analysis'],
                    'strengths': ['Content received and processed'],
                    'impact': 'Please try again or use alternative analysis method'
                },
                'analysis_method': 'fallback'
            }
        }