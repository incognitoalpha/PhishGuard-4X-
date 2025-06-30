import numpy as np
import pandas as pd
import re
import urllib.parse
import socket
import ssl
import requests
from datetime import datetime, timedelta
import whois
import joblib
import warnings
from collections import Counter
import heapq
from bs4 import BeautifulSoup
import tldextract
import time
from Model import (
    HuffmanEncoder, 
    HuffmanNode, 
    OptimizedPhishingDetector, 
    PhishingFeatureExtractor
)

warnings.filterwarnings('ignore')

class URLFeatureExtractor:
    """Extract phishing detection features from URLs"""
    
    def __init__(self):
        # These feature names must match exactly with the model's expected features
        self.feature_names = [
            'URL_Length', 'Shortining_Service', 'having_At_Symbol',
            'double_slash_redirecting', 'Prefix_Suffix', 'having_Sub_Domain', 'SSLfinal_State',
            'Domain_registeration_length', 'Favicon', 'port', 'HTTPS_token', 'Request_URL',
            'URL_of_Anchor', 'Links_in_tags', 'SFH', 'Submitting_to_email', 'Abnormal_URL',
            'Redirect', 'on_mouseover', 'RightClick', 'popUpWindow', 'Iframe',
            'age_of_domain', 'web_traffic', 'Page_Rank', 'Google_Index', 'Statistical_report'
        ]
        self.suspicious_domains = [
            'bit.ly',  'goo.gl', 't.co', 'ow.ly', 'is.gd',
            'buff.ly', 'adf.ly', 'bc.vc', 'buzurl.com', 'cli.gs', 'cur.lv'
        ]
        self.known_legitimate_domains = [
            'google.com', 'github.com', 'microsoft.com', 'amazon.com', 'facebook.com',
            'linkedin.com', 'twitter.com', 'instagram.com',  'wikipedia.org',
            'stackoverflow.com', 'medium.com', 'reddit.com', 'quora.com', 'takeuforward.org'
        ]

    def extract_all_features(self, url):
        """Extract all features from a URL"""
        features = {}
        
        # Initialize all features with default values
        for feature in self.feature_names:
            features[feature] = 0
            
        try:
            # Parse URL
            parsed_url = urllib.parse.urlparse(url)
            domain = parsed_url.netloc
            path = parsed_url.path
            
            # Check if domain is in known lists
            domain_parts = domain.split('.')
            base_domain = '.'.join(domain_parts[-2:]) if len(domain_parts) > 1 else domain
            
            # URL Length
            features['URL_Length'] = len(url)
            
            # Shortening Service
            features['Shortining_Service'] = 1 if any(shortener in domain for shortener in self.suspicious_domains) else 0
            
            # @ Symbol
            features['having_At_Symbol'] = 1 if '@' in url else 0
            
            # Double Slash
            features['double_slash_redirecting'] = 1 if '//' in url[8:] else 0
            
            # Prefix-Suffix
            features['Prefix_Suffix'] = 1 if '-' in domain else 0
            
            # Sub Domain
            features['having_Sub_Domain'] = 1 if len(domain_parts) > 2 and base_domain not in self.known_legitimate_domains else 0
            
            # SSL State
            try:
                context = ssl.create_default_context()
                with socket.create_connection((domain, 443)) as sock:
                    with context.wrap_socket(sock, server_hostname=domain) as ssock:
                        features['SSLfinal_State'] = 1  # Valid SSL
            except:
                features['SSLfinal_State'] = -1  # No SSL
            
            # Domain Registration Length
            try:
                w = whois.whois(domain)
                if w.creation_date:
                    if isinstance(w.creation_date, list):
                        creation_date = w.creation_date[0]
                    else:
                        creation_date = w.creation_date
                    age = (datetime.now() - creation_date).days
                    features['Domain_registeration_length'] = age
                    features['age_of_domain'] = age
            except:
                features['Domain_registeration_length'] = -1
                features['age_of_domain'] = -1
            
            # Port
            features['port'] = 1 if ':' in domain else 0
            
            # HTTPS Token
            features['HTTPS_token'] = 1 if 'https' in domain.lower() else 0
            
            # Request URL (simplified)
            features['Request_URL'] = 0  # Default to neutral
            
            # URL of Anchor (simplified)
            features['URL_of_Anchor'] = 0  # Default to neutral
            
            # Links in Tags (simplified)
            features['Links_in_tags'] = 0  # Default to neutral
            
            # SFH (Server Form Handler)
            features['SFH'] = 0  # Default to neutral
            
            # Submitting to Email
            features['Submitting_to_email'] = 1 if 'mailto:' in url.lower() else 0
            
            # Abnormal URL
            features['Abnormal_URL'] = 1 if len(path.split('/')) > 4 else 0
            
            # Redirect
            features['Redirect'] = 0  # Default to neutral
            
            # Mouse Over
            features['on_mouseover'] = 0  # Default to neutral
            
            # Right Click
            features['RightClick'] = 0  # Default to neutral
            
            # Popup Window
            features['popUpWindow'] = 0  # Default to neutral
            
            # Iframe
            features['Iframe'] = 0  # Default to neutral
            
            # Web Traffic (simplified)
            features['web_traffic'] = 1 if base_domain in self.known_legitimate_domains else 0
            
            # Page Rank (simplified)
            features['Page_Rank'] = 1 if base_domain in self.known_legitimate_domains else 0
            
            # Google Index (simplified)
            features['Google_Index'] = 1 if base_domain in self.known_legitimate_domains else 0
            
            # Statistical Report (simplified)
            features['Statistical_report'] = 1 if base_domain in self.known_legitimate_domains else 0
            
        except Exception as e:
            print(f"Error extracting features: {e}")
            # Return default features if extraction fails
            return {feature: 0 for feature in self.feature_names}
            
        return features

class PhishingURLTester:
    """Test interface for URL phishing detection"""
    
    def __init__(self, model_path='optimized_phishing_detector.joblib', preloaded_model=None):
        self.feature_extractor = URLFeatureExtractor()
        self.model = None
        self.model_path = model_path
        
        if preloaded_model is not None:
            # Use preloaded model if provided
            self.model = preloaded_model
            print("✓ Model preloaded successfully!")
        else:
            # Otherwise, load the model from path
            self.load_model()

        # Define thresholds for different risk levels
        self.thresholds = {
            'high_risk': 0.75,    # 75% or higher probability of phishing
            'medium_risk': 0.60,  # 60% or higher probability of phishing
            'low_risk': 0.40      # 40% or higher probability of phishing
        }
        # Add more known legitimate domains
        self.feature_extractor.known_legitimate_domains.extend([
            'nptel.ac.in',  # NPTEL domain
            'iit.ac.in',    # IIT domains
            'iisc.ac.in',   # IISc domain
            'gov.in',       # Indian government domains
            'edu.in',       # Indian educational domains
            'nic.in',       # National Informatics Centre
            'ac.in'         # Academic institutions
        ])
    
    def load_model(self):
        """Load the trained model"""
        try:
            # Load the model and ensure it's an instance of OptimizedPhishingDetector
            loaded_model = joblib.load(self.model_path)
            if isinstance(loaded_model, OptimizedPhishingDetector):
                self.model = loaded_model
                print("✓ Model loaded successfully!")
            else:
                print("✗ Error: Loaded model is not of the correct type")
                self.model = None
        except FileNotFoundError:
            print(f"✗ Model file '{self.model_path}' not found!")
            print("Please run the training script first to create the model.")
            self.model = None
        except Exception as e:
            print(f"✗ Error loading model: {e}")
            self.model = None
    
    def analyze_url(self, url):
        """Analyze a URL and return prediction with detailed report"""
        if self.model is None:
            return None
        
        print("="*80)
        print(f"PHISHING URL ANALYSIS REPORT")
        print("="*80)
        
        # Validate URL
        if not self._validate_url(url):
            print(f"Invalid URL format: {url}")
            return None
        
        # Extract features
        print(f"\n1. EXTRACTING FEATURES...")
        print("-"*40)
        features = self.feature_extractor.extract_all_features(url)
        
        # Create DataFrame for prediction
        feature_df = pd.DataFrame([features])
        
        # Ensure all required features are present
        for feature in self.feature_extractor.feature_names:
            if feature not in feature_df.columns:
                feature_df[feature] = 0
        
        # Reorder columns to match training data
        feature_df = feature_df[self.feature_extractor.feature_names]
        
        # Make prediction
        print(f"\n2. MAKING PREDICTION...")
        print("-"*40)
        
        try:
            prediction = self.model.predict(feature_df)[0]
            probabilities = self.model.predict_proba(feature_df)[0]
            
            # First generate the detailed feature report to get risk levels
            print("\n3. DETAILED FEATURE ANALYSIS")
            print("="*80)
            print(f"{'Feature':<20} {'Value':<15} {'Interpretation':<20} {'Risk Level':<10}")
            print("-"*80)
            
            high_risk_features = []
            for feature, value in features.items():
                interpretation = self._get_feature_interpretation(feature, value)
                risk_level = self._get_risk_level(feature, value)
                print(f"{feature:<20} {str(value):<15} {interpretation:<20} {risk_level:<10}")
                if risk_level == 'HIGH':
                    high_risk_features.append(feature)
            
            print(f"\nNumber of high-risk features: {len(high_risk_features)}")
            
            # Determine if URL is phishing based on high risk features
            is_phishing = len(high_risk_features) >= 2  # Only mark as phishing if 2 or more high-risk features
            
            # Get prediction result
            confidence = max(probabilities) * 100
            
            print(f"\nPrediction: {'PHISHING' if is_phishing else 'LEGITIMATE'}")
            print(f"Confidence: {confidence:.1f}%")
            print(f"Phishing Probability: {probabilities[0]*100:.1f}%")
            print(f"Legitimate Probability: {probabilities[1]*100:.1f}%")
            
            if high_risk_features:
                print("\nHigh Risk Features Detected:")
                for feature in high_risk_features:
                    print(f"- {feature}")
            
            # Generate risk assessment
            self._generate_risk_assessment(features, is_phishing, confidence)
            
            return {
                'url': url,
                'prediction': 'PHISHING' if is_phishing else 'LEGITIMATE',
                'confidence': confidence,
                'phishing_probability': probabilities[0] * 100,
                'legitimate_probability': probabilities[1] * 100,
                'features': features,
                'is_phishing': is_phishing,
                'high_risk_features': high_risk_features
            }
            
        except Exception as e:
            print(f"Error during prediction: {e}")
            return None
    
    def _validate_url(self, url):
        """Validate URL format"""
        if not url.startswith(('http://', 'https://')):
            return False
        
        try:
            result = urllib.parse.urlparse(url)
            return all([result.scheme, result.netloc])
        except:
            return False
    
    def _generate_feature_report(self, features, feature_series, is_phishing):
        """Generate detailed feature analysis report"""
        print(f"\n3. DETAILED FEATURE ANALYSIS")
        print("="*80)
        
        # Feature descriptions and interpretations
        feature_descriptions = {
            'URL_Length': ('URL Length', 'characters', lambda x: 'Long' if x > 75 else 'Medium' if x > 30 else 'Short'),
            'Shortining_Service': ('URL Shortening Service', '', lambda x: 'Detected' if x == 1 else 'Not Detected'),
            'having_At_Symbol': ('@ Symbol in URL', '', lambda x: 'Present' if x == 1 else 'Absent'),
            'double_slash_redirecting': ('Double Slash Redirecting', '', lambda x: 'Detected' if x == 1 else 'Not Detected'),
            'Prefix_Suffix': ('Prefix-Suffix in Domain', '', lambda x: 'Present' if x == 1 else 'Absent'),
            'having_Sub_Domain': ('Subdomains', '', lambda x: 'Multiple' if x == 1 else 'Single/None'),
            'SSLfinal_State': ('SSL Certificate', '', lambda x: 'Valid' if x == 1 else 'Invalid/Expired' if x == 0 else 'No SSL'),
            'Domain_registeration_length': ('Domain Registration', 'days', lambda x: 'Long-term' if x > 365 else 'Short-term' if x > 0 else 'Unknown'),
            'port': ('Port Usage', '', lambda x: 'Suspicious' if x == 1 else 'Standard'),
            'HTTPS_token': ('HTTPS in Domain Name', '', lambda x: 'Misleading' if x == 1 else 'Normal'),
            'Request_URL': ('External Resources', '', lambda x: 'Low' if x == 1 else 'Medium' if x == 0 else 'High'),
            'URL_of_Anchor': ('External Anchors', '', lambda x: 'Low' if x == 1 else 'Medium' if x == 0 else 'High'),
            'SFH': ('Server Form Handler', '', lambda x: 'Safe' if x == 1 else 'Suspicious' if x == 0 else 'Dangerous'),
            'Abnormal_URL': ('URL Structure', '', lambda x: 'Normal' if x == 1 else 'Abnormal'),
            'Redirect': ('URL Redirects', '', lambda x: 'Normal' if x == 1 else 'Multiple' if x == 0 else 'Excessive'),
            'age_of_domain': ('Domain Age', 'days', lambda x: 'Old' if x > 365 else 'Recent' if x > 0 else 'Unknown'),
            'web_traffic': ('Web Traffic Rank', '', lambda x: 'High' if x == 1 else 'Medium' if x == 0 else 'Low'),
            'Page_Rank': ('Page Rank', '', lambda x: 'High' if x == 1 else 'Low'),
            'Google_Index': ('Google Indexing', '', lambda x: 'Indexed' if x == 1 else 'Not Indexed'),
            'Statistical_report': ('Statistical Reports', '', lambda x: 'Clean' if x == 1 else 'Suspicious')
        }
        
        # Categorize features
        suspicious_features = []
        safe_features = []
        neutral_features = []
        
        print(f"{'Feature':<25} {'Value':<15} {'Interpretation':<20} {'Risk Level':<15}")
        print("-" * 85)
        
        for feature, value in features.items():
            if feature in feature_descriptions:
                desc, unit, interpret_func = feature_descriptions[feature]
                interpretation = interpret_func(value)
                
                # Determine risk level based on feature and value
                risk_level = self._assess_feature_risk(feature, value)
                
                print(f"{desc:<25} {str(value) + ' ' + unit:<15} {interpretation:<20} {risk_level:<15}")
                
                if risk_level == 'HIGH':
                    suspicious_features.append((desc, interpretation))
                elif risk_level == 'LOW':
                    safe_features.append((desc, interpretation))
                else:
                    neutral_features.append((desc, interpretation))
        
        # Summary of findings
        print(f"\n4. FEATURE SUMMARY")
        print("-" * 40)
        
        if suspicious_features:
            print(f"\n🚨 SUSPICIOUS INDICATORS ({len(suspicious_features)}):")
            for i, (feature, interpretation) in enumerate(suspicious_features, 1):
                print(f"   {i}. {feature}: {interpretation}")
        
        if safe_features:
            print(f"\n✅ POSITIVE INDICATORS ({len(safe_features)}):")
            for i, (feature, interpretation) in enumerate(safe_features, 1):
                print(f"   {i}. {feature}: {interpretation}")
        
        if neutral_features:
            print(f"\n⚪ NEUTRAL INDICATORS ({len(neutral_features)}):")
            for i, (feature, interpretation) in enumerate(neutral_features, 1):
                print(f"   {i}. {feature}: {interpretation}")
    
    def _assess_feature_risk(self, feature, value):
        """Assess risk level for individual features"""
        high_risk_conditions = {
            'URL_Length': lambda x: x > 100,
            'Shortining_Service': lambda x: x == 1,
            'having_At_Symbol': lambda x: x == 1,
            'double_slash_redirecting': lambda x: x == 1,
            'Prefix_Suffix': lambda x: x == 1,
            'having_Sub_Domain': lambda x: x == 1,
            'SSLfinal_State': lambda x: x == -1,
            'Domain_registeration_length': lambda x: x < 30 and x > 0,
            'port': lambda x: x == 1,
            'HTTPS_token': lambda x: x == 1,
            'Request_URL': lambda x: x == -1,
            'URL_of_Anchor': lambda x: x == -1,
            'SFH': lambda x: x == -1,
            'Abnormal_URL': lambda x: x == -1,
            'Redirect': lambda x: x == -1,
            'age_of_domain': lambda x: x < 30 and x > 0,
            'web_traffic': lambda x: x == -1,
            'Page_Rank': lambda x: x == -1,
            'Google_Index': lambda x: x == -1,
            'Statistical_report': lambda x: x == -1
        }
        
        low_risk_conditions = {
            'URL_Length': lambda x: x < 30,
            'SSLfinal_State': lambda x: x == 1,
            'Domain_registeration_length': lambda x: x > 365,
            'age_of_domain': lambda x: x > 365,
            'web_traffic': lambda x: x == 1,
            'Page_Rank': lambda x: x == 1,
            'Google_Index': lambda x: x == 1,
            'Statistical_report': lambda x: x == 1
        }
        
        if feature in high_risk_conditions and high_risk_conditions[feature](value):
            return 'HIGH'
        elif feature in low_risk_conditions and low_risk_conditions[feature](value):
            return 'LOW'
        else:
            return 'MEDIUM'
    
    def _generate_risk_assessment(self, features, is_phishing, confidence):
        """Generate overall risk assessment and recommendations"""
        print(f"\n5. RISK ASSESSMENT & RECOMMENDATIONS")
        print("="*80)
        
        if is_phishing:
            print("🚨 HIGH RISK - POTENTIAL PHISHING WEBSITE")
            print("\nRECOMMENDations:")
            print("❌ DO NOT enter personal information, passwords, or financial details")
            print("❌ DO NOT download files from this website")
            print("❌ DO NOT click on suspicious links")
            print("✅ Verify the website URL carefully")
            print("✅ Contact the organization directly through official channels")
            print("✅ Report this website to anti-phishing authorities")
        else:
            if confidence > 80:
                print("✅ LOW RISK - LIKELY LEGITIMATE WEBSITE")
                print("\nGeneral Security Tips:")
                print("✅ Always verify SSL certificates")
                print("✅ Check for secure payment methods")
                print("✅ Be cautious with personal information")
            else:
                print("⚠️  MODERATE RISK - PROCEED WITH CAUTION")
                print("\nRECOMMENDations:")
                print("⚠️  Exercise caution when entering sensitive information")
                print("✅ Verify the website's authenticity through other means")
                print("✅ Check user reviews and ratings")
                print("✅ Ensure secure connection (HTTPS)")
        
        # Additional security recommendations
        print(f"\n📋 GENERAL SECURITY CHECKLIST:")
        print("□ Verify the URL spelling and domain")
        print("□ Check for HTTPS and valid SSL certificate")
        print("□ Look for contact information and privacy policy")
        print("□ Check website design quality and professionalism")
        print("□ Verify through official company websites or contacts")
        print("□ Use updated antivirus and browser security features")

    def _get_risk_level(self, feature, value):
        """Get risk level for a feature value"""
        risk_levels = {
            'URL_Length': {
                'high': 'HIGH',
                'medium': 'MEDIUM',
                'low': 'LOW'
            },
            'Shortining_Service': {
                1: 'HIGH',
                0: 'LOW'  # Changed back from MEDIUM to LOW
            },
            'having_At_Symbol': {
                1: 'HIGH',
                0: 'LOW'  # Changed back from MEDIUM to LOW
            },
            'double_slash_redirecting': {
                1: 'HIGH',
                0: 'LOW'  # Changed back from MEDIUM to LOW
            },
            'Prefix_Suffix': {
                1: 'HIGH',
                0: 'LOW'  # Changed back from MEDIUM to LOW
            },
            'having_Sub_Domain': {
                1: 'HIGH',
                0: 'LOW'  # Changed back from MEDIUM to LOW
            },
            'SSLfinal_State': {
                1: 'LOW',  # Valid SSL is low risk
                0: 'HIGH', # Invalid SSL is high risk
                -1: 'HIGH' # No SSL is high risk
            },
            'Domain_registeration_length': {
                'high': 'LOW',
                'medium': 'MEDIUM',
                'low': 'HIGH'
            },
            'port': {
                1: 'HIGH',
                0: 'LOW'  # Changed back from MEDIUM to LOW
            },
            'HTTPS_token': {
                1: 'HIGH',
                0: 'LOW'  # Changed back from MEDIUM to LOW
            },
            'Request_URL': {
                1: 'HIGH',
                0: 'LOW'  # Changed back from MEDIUM to LOW
            },
            'URL_of_Anchor': {
                1: 'HIGH',
                0: 'LOW'  # Changed back from MEDIUM to LOW
            },
            'SFH': {
                1: 'HIGH',
                0: 'LOW'  # Changed back from MEDIUM to LOW
            },
            'Abnormal_URL': {
                1: 'HIGH',
                0: 'LOW'  # Changed back from MEDIUM to LOW
            },
            'Redirect': {
                1: 'HIGH',
                0: 'LOW'  # Changed back from MEDIUM to LOW
            },
            'web_traffic': {
                1: 'LOW',
                0: 'LOW'  # Changed back from MEDIUM to LOW
            },
            'Page_Rank': {
                1: 'LOW',
                0: 'LOW'  # Changed back from MEDIUM to LOW
            },
            'Google_Index': {
                1: 'LOW',
                0: 'LOW'  # Changed back from MEDIUM to LOW
            },
            'Statistical_report': {
                1: 'LOW',
                0: 'LOW'  # Changed back from MEDIUM to LOW
            }
        }
        
        if feature in risk_levels:
            if isinstance(risk_levels[feature], dict):
                if feature in ['URL_Length', 'Domain_registeration_length']:  # Removed age_of_domain
                    if value > 100:
                        return risk_levels[feature]['high']
                    elif value > 50:
                        return risk_levels[feature]['medium']
                    else:
                        return risk_levels[feature]['low']
                else:
                    return risk_levels[feature].get(value, 'MEDIUM')
        return 'MEDIUM'

    def _get_feature_interpretation(self, feature, value):
        """Get human-readable interpretation of a feature value"""
        interpretations = {
            'URL_Length': {
                'high': 'Very Long (>100)',
                'medium': 'Medium (50-100)',
                'low': 'Short (<50)'
            },
            'Shortining_Service': {
                1: 'Detected',
                0: 'Not Detected'
            },
            'having_At_Symbol': {
                1: 'Present',
                0: 'Absent'
            },
            'double_slash_redirecting': {
                1: 'Detected',
                0: 'Not Detected'
            },
            'Prefix_Suffix': {
                1: 'Present',
                0: 'Absent'
            },
            'having_Sub_Domain': {
                1: 'Multiple',
                0: 'Single'
            },
            'SSLfinal_State': {
                1: 'Valid',
                0: 'Invalid',
                -1: 'No SSL'
            },
            'Domain_registeration_length': {
                'high': 'Long-term (>365 days)',
                'medium': 'Medium (30-365 days)',
                'low': 'Short-term (<30 days)'
            },
            'port': {
                1: 'Non-standard',
                0: 'Standard'
            },
            'HTTPS_token': {
                1: 'Suspicious',
                0: 'Normal'
            },
            'Request_URL': {
                1: 'External',
                0: 'Internal'
            },
            'URL_of_Anchor': {
                1: 'External',
                0: 'Internal'
            },
            'SFH': {
                1: 'Suspicious',
                0: 'Normal'
            },
            'Abnormal_URL': {
                1: 'Abnormal',
                0: 'Normal'
            },
            'Redirect': {
                1: 'Multiple',
                0: 'None'
            },
            'age_of_domain': {
                'high': 'Old (>365 days)',
                'medium': 'Medium (30-365 days)',
                'low': 'New (<30 days)'
            },
            'web_traffic': {
                1: 'High',
                0: 'Low'
            },
            'Page_Rank': {
                1: 'High',
                0: 'Low'
            },
            'Google_Index': {
                1: 'Indexed',
                0: 'Not Indexed'
            },
            'Statistical_report': {
                1: 'Clean',
                0: 'Suspicious'
            }
        }
        
        if feature in interpretations:
            if isinstance(interpretations[feature], dict):
                if feature in ['URL_Length', 'Domain_registeration_length']:
                    if value > 100:
                        return interpretations[feature]['high']
                    elif value > 50:
                        return interpretations[feature]['medium']
                    else:
                        return interpretations[feature]['low']
                else:
                    return interpretations[feature].get(value, 'Unknown')
        return 'Unknown'

def main():
    """Main function for interactive URL testing"""
    print("="*80)
    print("🔍 PHISHING URL DETECTION SYSTEM")
    print("="*80)
    print("This system analyzes URLs for potential phishing indicators.")
    print("It extracts various features and uses machine learning for detection.")
    print()
    
    # Initialize tester
    tester = PhishingURLTester()
    
    if tester.model is None:
        print("❌ Model not available. Please train the model first using the training script.")
        return
    
    print("✅ System ready for URL analysis!")
    print("\nInstructions:")
    print("- Enter a complete URL (including http:// or https://)")
    print("- Type 'quit' to exit")
    print("- Type 'examples' to see sample URLs for testing")
    print()
    
    while True:
        try:
            user_input = input("🌐 Enter URL to analyze: ").strip()
            
            if user_input.lower() == 'quit':
                print("👋 Thank you for using the Phishing Detection System!")
                break
                
            elif user_input.lower() == 'examples':
                print("\n📝 Example URLs for testing:")
                print("Legitimate websites:")
                print("  - https://www.google.com")
                print("  - https://www.github.com")
                print("  - https://www.amazon.com")
                print("\nSuspicious patterns to test:")
                print("  - URLs with @ symbols")
                print("  - Very long URLs")
                print("  - URLs with suspicious subdomains")
                print("  - Non-HTTPS banking/financial sites")
                print()
                continue
                
            elif user_input == '':
                continue
            
            # Analyze the URL
            result = tester.analyze_url(user_input)
            
            if result:
                print(f"\n{'='*80}")
                print(f"ANALYSIS COMPLETE")
                print(f"{'='*80}")
                print(f"Final Result: {result['prediction']}")
                print(f"Confidence: {result['confidence']:.1f}%")
                
                if result['is_phishing']:
                    print("🚨 WARNING: This URL shows signs of being a phishing website!")
                else:
                    print("✅ This URL appears to be legitimate.")
            
                print(f"\n{'='*80}")
            
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ An error occurred: {e}")
            print("Please try again with a valid URL.")

if __name__ == "__main__":
    main()