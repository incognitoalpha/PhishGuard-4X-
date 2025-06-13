from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import sys
import os

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from claudemodifiedensemble import (
    OptimizedPhishingDetector,
    PhishingFeatureExtractor,
    HuffmanEncoder,
    HuffmanNode
)
from claudemodifiedensembletesting import PhishingURLTester

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Initialize the phishing tester with the model path
model_path = 'optimized_phishing_detector.joblib'
tester = PhishingURLTester(model_path=model_path)

@app.route('/analyze', methods=['POST'])
def analyze_url():
    data = request.get_json()
    url = data.get('url')
    
    if not url:
        return jsonify({'error': 'No URL provided'}), 400
    
    try:
        # Analyze the URL using the Python implementation
        result = tester.analyze_url(url)
        
        if result is None:
            return jsonify({'error': 'Invalid URL format'}), 400
        
        # Format the response
        response = {
            'url': result['url'],
            'prediction': result['prediction'],
            'confidence': result['confidence'],
            'phishing_probability': result['phishing_probability'],
            'legitimate_probability': result['legitimate_probability'],
            'is_phishing': result['is_phishing'],
            'high_risk_features': result['high_risk_features'],
            'features': result['features']
        }
        
        return jsonify(response)
    except Exception as e:
        print(f"Error analyzing URL: {str(e)}")
        return jsonify({'error': f'Analysis failed: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(port=5000, debug=True) 