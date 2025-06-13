import numpy as np
import pandas as pd
from collections import Counter
import heapq
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_recall_fscore_support, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
import warnings
import joblib
warnings.filterwarnings('ignore')

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("Warning: XGBoost not available. Using GradientBoostingClassifier as substitute.")

class HuffmanNode:
    def __init__(self, char: str, freq: int, left=None, right=None):
        self.char = char
        self.freq = freq
        self.left = left
        self.right = right
    
    def __lt__(self, other):
        return self.freq < other.freq

class HuffmanEncoder:
    def __init__(self):
        self.codes = {}
        self.root = None
    
    def build_tree(self, text: str) -> None:
        """Build Huffman tree from text frequency"""
        if not text:
            return
        
        # Count character frequencies
        freq = Counter(text)
        
        # Create priority queue with leaf nodes
        heap = [HuffmanNode(char, count) for char, count in freq.items()]
        heapq.heapify(heap)
        
        # Build tree
        while len(heap) > 1:
            left = heapq.heappop(heap)
            right = heapq.heappop(heap)
            
            merged = HuffmanNode(None, left.freq + right.freq, left, right)
            heapq.heappush(heap, merged)
        
        self.root = heap[0] if heap else None
        self._generate_codes(self.root, "")
    
    def _generate_codes(self, node: HuffmanNode, code: str) -> None:
        """Generate Huffman codes for each character"""
        if node is None:
            return
        
        if node.char is not None:  # Leaf node
            self.codes[node.char] = code if code else "0"
            return
        
        self._generate_codes(node.left, code + "0")
        self._generate_codes(node.right, code + "1")
    
    def encode(self, text: str) -> str:
        """Encode text using Huffman codes"""
        return ''.join(self.codes.get(char, '') for char in text)

class PhishingFeatureExtractor:
    def __init__(self):
        self.huffman_encoder = HuffmanEncoder()
        self.scaler = StandardScaler()
        # Define most important features for phishing detection based on research
        self.important_features = [
            'URL_Length', 'Shortining_Service', 'having_At_Symbol',
            'double_slash_redirecting', 'Prefix_Suffix', 'having_Sub_Domain', 
            'SSLfinal_State', 'Domain_registeration_length', 'port', 
            'HTTPS_token', 'Request_URL', 'URL_of_Anchor', 'SFH', 
            'Abnormal_URL', 'Redirect', 'age_of_domain', 'web_traffic', 
            'Page_Rank', 'Google_Index', 'Statistical_report'
        ]
        
    def extract_huffman_features(self, data_str: str) -> dict:
        """Extract key Huffman-based features from data string representation"""
        if not data_str:
            return {
                'huffman_compression_ratio': 0,
                'huffman_bit_ratio': 0,
                'huffman_pattern_complexity': 0,
                'huffman_avg_code_length': 0,
                'huffman_weighted_avg_length': 0
            }
        
        # Build Huffman tree and encode
        self.huffman_encoder.build_tree(data_str)
        encoded = self.huffman_encoder.encode(data_str)
        
        if not encoded:
            return {
                'huffman_compression_ratio': 0,
                'huffman_bit_ratio': 0,
                'huffman_pattern_complexity': 0,
                'huffman_avg_code_length': 0,
                'huffman_weighted_avg_length': 0
            }
        
        # Key Huffman features
        huffman_features = {
            'huffman_compression_ratio': len(encoded) / len(data_str),
            'huffman_bit_ratio': encoded.count('1') / len(encoded),
        }
        
        # Pattern complexity (entropy-like measure)
        chunks = [encoded[i:i+4] for i in range(0, len(encoded), 4)]
        unique_chunks = len(set(chunks))
        huffman_features['huffman_pattern_complexity'] = unique_chunks / len(chunks) if chunks else 0
        
        # Huffman code statistics
        if self.huffman_encoder.codes:
            code_lengths = [len(code) for code in self.huffman_encoder.codes.values()]
            huffman_features['huffman_avg_code_length'] = np.mean(code_lengths)
            
            # Weighted average length
            char_freq = Counter(data_str)
            total_chars = len(data_str)
            weighted_avg = 0
            
            for char, freq in char_freq.items():
                if char in self.huffman_encoder.codes:
                    weight = freq / total_chars
                    code_len = len(self.huffman_encoder.codes[char])
                    weighted_avg += weight * code_len
            
            huffman_features['huffman_weighted_avg_length'] = weighted_avg
        else:
            huffman_features['huffman_avg_code_length'] = 0
            huffman_features['huffman_weighted_avg_length'] = 0
        
        return huffman_features
    
    def prepare_features(self, X: pd.DataFrame) -> np.ndarray:
        """Prepare enhanced feature matrix with selected important features"""
        # Select only important features that exist in the dataset
        available_features = [f for f in self.important_features if f in X.columns]
        X_selected = X[available_features].copy()
        
        print(f"Using {len(available_features)} important features: {available_features}")
        
        # Add Huffman-based features for each row
        huffman_features_list = []
        for i in range(len(X_selected)):
            # Convert row to string representation
            row_str = ' '.join(str(val) for val in X_selected.iloc[i].values)
            huffman_features = self.extract_huffman_features(row_str)
            huffman_features_list.append(huffman_features)
        
        # Convert to DataFrame and combine with selected features
        huffman_df = pd.DataFrame(huffman_features_list, index=X_selected.index)
        enhanced_features = pd.concat([X_selected, huffman_df], axis=1)
        
        return enhanced_features.fillna(0).values

class OptimizedPhishingDetector:
    def __init__(self):
        self.feature_extractor = PhishingFeatureExtractor()
        self.models = {}
        self.ensemble_model = None
        self.feature_selector = None
        self.is_trained = False
        
    def _initialize_models(self):
        """Initialize top 4 performing models for the ensemble"""
        models = {}
        
        # 1. Random Forest - Generally excellent for phishing detection
        models['RandomForest'] = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42
        )
        
        # 2. XGBoost - High performance gradient boosting
        if XGBOOST_AVAILABLE:
            models['XGBoost'] = xgb.XGBClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                eval_metric='logloss'
            )
        else:
            from sklearn.ensemble import GradientBoostingClassifier
            models['XGBoost'] = GradientBoostingClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42
            )
        
        # 3. SVM - Excellent for binary classification with good generalization
        models['SVM'] = SVC(
            probability=True, 
            kernel='rbf', 
            C=10, 
            gamma='scale',
            random_state=42
        )
        
        # 4. Logistic Regression - Simple, interpretable, good baseline
        models['LogisticRegression'] = LogisticRegression(
            random_state=42, 
            max_iter=1000, 
            C=10,
            penalty='l2'
        )
        
        return models
    
    def train(self, X: pd.DataFrame, y: pd.Series, use_cross_validation=True, feature_selection=True):
        """Train selected models and create ensemble"""
        print("Preparing enhanced features with reduced feature set...")
        
        # Handle target variable encoding (convert -1 to 0 for binary classification)
        y_binary = y.copy()
        y_binary[y_binary == -1] = 0  # Convert -1 (phishing) to 0, keep 1 (legitimate) as 1
        
        # Prepare features
        X_enhanced = self.feature_extractor.prepare_features(X)
        
        print(f"Enhanced feature matrix shape: {X_enhanced.shape}")
        
        # Scale features
        X_scaled = self.feature_extractor.scaler.fit_transform(X_enhanced)
        
        # Optional feature selection - select top features
        if feature_selection:
            print("Performing additional feature selection...")
            n_features = min(X_scaled.shape[1], 20)  # Select top 20 features
            self.feature_selector = SelectKBest(score_func=f_classif, k=n_features)
            X_selected = self.feature_selector.fit_transform(X_scaled, y_binary)
            print(f"Selected {n_features} features out of {X_scaled.shape[1]}")
        else:
            X_selected = X_scaled
            
        # Initialize models
        self.models = self._initialize_models()
        
        # Train individual models and evaluate
        print(f"\nTraining {len(self.models)} optimized models...")
        model_scores = {}
        trained_models = []
        
        for name, model in self.models.items():
            try:
                print(f"Training {name}...")
                
                # Cross-validation
                if use_cross_validation:
                    cv_scores = cross_val_score(
                        model, X_selected, y_binary,
                        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
                        scoring='accuracy'
                    )
                    model_scores[name] = {
                        'cv_mean': cv_scores.mean(),
                        'cv_std': cv_scores.std()
                    }
                    print(f"  CV Score: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
                
                # Train model
                model.fit(X_selected, y_binary)
                train_score = model.score(X_selected, y_binary)
                model_scores[name]['train_score'] = train_score
                
                trained_models.append((name, model))
                
            except Exception as e:
                print(f"  Error training {name}: {e}")
                continue
        
        # Create ensemble with all trained models (max 4)
        if len(trained_models) >= 2:
            print(f"\nCreating ensemble with {len(trained_models)} models...")
            
            # Sort models by CV performance
            if use_cross_validation:
                sorted_models = sorted(trained_models, 
                                     key=lambda x: model_scores[x[0]]['cv_mean'], 
                                     reverse=True)
            else:
                sorted_models = sorted(trained_models, 
                                     key=lambda x: model_scores[x[0]]['train_score'], 
                                     reverse=True)
            
            # Use all available models (maximum 4)
            ensemble_models = sorted_models
            
            # Create weighted ensemble based on CV scores
            weights = None
            if use_cross_validation:
                weights = [model_scores[name]['cv_mean'] for name, _ in ensemble_models]
            
            self.ensemble_model = VotingClassifier(
                estimators=ensemble_models,
                voting='soft',
                weights=weights
            )
            
            self.ensemble_model.fit(X_selected, y_binary)
            ensemble_score = self.ensemble_model.score(X_selected, y_binary)
            
            print(f"Ensemble training score: {ensemble_score:.3f}")
            print(f"Ensemble models: {[name for name, _ in ensemble_models]}")
            
        else:
            print("Not enough models trained successfully for ensemble!")
            return None
        
        self.is_trained = True
        
        return {
            'individual_scores': model_scores,
            'ensemble_score': ensemble_score,
            'n_features': X_selected.shape[1],
            'ensemble_models': [name for name, _ in ensemble_models],
            'selected_features': self.feature_extractor.important_features
        }
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions using the ensemble model"""
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        
        # Prepare features
        X_enhanced = self.feature_extractor.prepare_features(X)
        X_scaled = self.feature_extractor.scaler.transform(X_enhanced)
        
        if self.feature_selector:
            X_selected = self.feature_selector.transform(X_scaled)
        else:
            X_selected = X_scaled
        
        # Make predictions (convert back to original labels)
        predictions = self.ensemble_model.predict(X_selected)
        predictions[predictions == 0] = -1  # Convert 0 back to -1 for phishing
        
        return predictions
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Get prediction probabilities"""
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        
        # Prepare features
        X_enhanced = self.feature_extractor.prepare_features(X)
        X_scaled = self.feature_extractor.scaler.transform(X_enhanced)
        
        if self.feature_selector:
            X_selected = self.feature_selector.transform(X_scaled)
        else:
            X_selected = X_scaled
        
        return self.ensemble_model.predict_proba(X_selected)
    
    def get_feature_importance(self) -> dict:
        """Get feature importance from models that support it"""
        importance_dict = {}
        
        for name, model in self.ensemble_model.named_estimators_.items():
            if hasattr(model, 'feature_importances_'):
                importance_dict[name] = model.feature_importances_
            elif hasattr(model, 'coef_'):
                importance_dict[name] = np.abs(model.coef_[0])
        
        return importance_dict

def load_and_preprocess_data(file_path=None, sample_data=None):
    """Load and preprocess the phishing dataset"""
    if sample_data is not None:
        # Use provided sample data
        df = sample_data.copy()
    elif file_path is not None:
        # Load from file
        df = pd.read_csv(file_path)
    else:
        # Create sample data for demonstration
        print("Creating sample dataset for demonstration...")
        np.random.seed(42)
        
        # Create sample data with most important features for phishing detection
        n_samples = 1000
        important_features = [
            'URL_Length', 'Shortining_Service', 'having_At_Symbol',
            'double_slash_redirecting', 'Prefix_Suffix', 'having_Sub_Domain', 
            'SSLfinal_State', 'Domain_registeration_length', 'port', 
            'HTTPS_token', 'Request_URL', 'URL_of_Anchor', 'SFH', 
            'Abnormal_URL', 'Redirect', 'age_of_domain', 'web_traffic', 
            'Page_Rank', 'Google_Index', 'Statistical_report'
        ]
        
        # Generate realistic data for features
        data = {}
        for feature in important_features:
            if feature in ['URL_Length', 'age_of_domain', 'Domain_registeration_length']:
                # Continuous features - make phishing URLs tend to be longer/shorter
                if feature == 'URL_Length':
                    data[feature] = np.random.randint(10, 200, n_samples)
                elif feature == 'age_of_domain':
                    data[feature] = np.random.randint(0, 365*10, n_samples)  # 0-10 years
                else:
                    data[feature] = np.random.randint(0, 365*2, n_samples)  # 0-2 years
            else:
                # Categorical features (-1, 0, 1)
                data[feature] = np.random.choice([-1, 0, 1], n_samples, p=[0.4, 0.2, 0.4])
        
        # Create target variable with realistic distribution
        data['Result'] = np.random.choice([1, -1], n_samples, p=[0.6, 0.4])  # 60% legitimate, 40% phishing
        
        df = pd.DataFrame(data)
    
    print(f"Dataset shape: {df.shape}")
    print(f"Target distribution:\n{df['Result'].value_counts()}")
    print(f"Features: {list(df.columns[:-1])}")
    
    # Separate features and target
    X = df.drop('Result', axis=1)
    y = df['Result']
    
    return X, y

def main():
    """Main function to demonstrate the optimized phishing detection system"""
    print("=== Optimized Phishing Detection with Reduced Features and 4-Model Ensemble ===\n")
    
    # Load and preprocess data
    print("Loading dataset...")
    X, y = load_and_preprocess_data(file_path="Phishing_Websites_Data.csv")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    
    # Initialize and train detector
    detector = OptimizedPhishingDetector()
    
    print("\n" + "="*60)
    training_results = detector.train(X_train, y_train, use_cross_validation=True)
    
    if training_results is None:
        print("Training failed!")
        return
    
    # Save the trained model
    print("\nSaving optimized model...")
    joblib.dump(detector, 'optimized_phishing_detector.joblib')
    print("Model saved as 'optimized_phishing_detector.joblib'")
    
    print("\n" + "="*60)
    print("TRAINING RESULTS")
    print("="*60)
    
    # Display model information
    print(f"\nSelected Important Features ({len(training_results['selected_features'])}):")
    print(", ".join(training_results['selected_features']))
    
    print(f"\nTotal Features Used (including Huffman): {training_results['n_features']}")
    print(f"Ensemble Models: {', '.join(training_results['ensemble_models'])}")
    
    # Display individual model performances
    print(f"\nModel Performance (Cross-Validation):")
    print("-" * 55)
    print(f"{'Model':<20} {'CV Score':<15} {'Train Score':<12}")
    print("-" * 55)
    
    for model_name, scores in training_results['individual_scores'].items():
        cv_score = f"{scores['cv_mean']:.3f} ± {scores['cv_std']:.3f}"
        train_score = f"{scores['train_score']:.3f}"
        print(f"{model_name:<20} {cv_score:<15} {train_score:<12}")
    
    print(f"\nEnsemble Training Score: {training_results['ensemble_score']:.3f}")
    
    # Test the model
    print("\n" + "="*60)
    print("TESTING RESULTS")
    print("="*60)
    
    # Make predictions
    y_pred = detector.predict(X_test)
    y_pred_proba = detector.predict_proba(X_test)
    
    # Convert for metrics calculation
    y_test_binary = y_test.copy()
    y_test_binary[y_test_binary == -1] = 0
    y_pred_binary = y_pred.copy()
    y_pred_binary[y_pred_binary == -1] = 0
    
    # Calculate metrics
    accuracy = accuracy_score(y_test_binary, y_pred_binary)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test_binary, y_pred_binary, average='weighted'
    )
    
    try:
        auc = roc_auc_score(y_test_binary, y_pred_proba[:, 1])
    except:
        auc = 0.0
    
    print(f"Test Accuracy: {accuracy:.3f}")
    print(f"Precision: {precision:.3f}")
    print(f"Recall: {recall:.3f}")
    print(f"F1-Score: {f1:.3f}")
    print(f"AUC: {auc:.3f}")
    
    # Detailed classification report
    print(f"\nDetailed Classification Report:")
    print("-" * 40)
    target_names = ['Phishing', 'Legitimate']
    print(classification_report(y_test_binary, y_pred_binary, target_names=target_names))
    
    # Confusion Matrix
    print(f"Confusion Matrix:")
    print("-" * 20)
    cm = confusion_matrix(y_test_binary, y_pred_binary)
    print(f"{'':>12} {'Pred Phish':<12} {'Pred Legit':<12}")
    print(f"{'True Phish':<12} {cm[0,0]:<12} {cm[0,1]:<12}")
    print(f"{'True Legit':<12} {cm[1,0]:<12} {cm[1,1]:<12}")
    
    # Feature importance analysis
    print(f"\n" + "="*60)
    print("FEATURE IMPORTANCE ANALYSIS")
    print("="*60)
    
    try:
        importance_dict = detector.get_feature_importance()
        
        if importance_dict:
            print("\nTop 10 Most Important Features by Model:")
            for model_name, importances in importance_dict.items():
                print(f"\n{model_name}:")
                indices = np.argsort(importances)[::-1][:10]
                
                # Create feature names (original + Huffman features)
                feature_names = (detector.feature_extractor.important_features + 
                               ['huffman_compression_ratio', 'huffman_bit_ratio', 
                                'huffman_pattern_complexity', 'huffman_avg_code_length', 
                                'huffman_weighted_avg_length'])
                
                for i, idx in enumerate(indices):
                    if idx < len(feature_names):
                        feature_name = feature_names[idx]
                        print(f"  {i+1}. {feature_name}: {importances[idx]:.3f}")
        else:
            print("Feature importance not available for current models.")
    except Exception as e:
        print(f"Error analyzing feature importance: {e}")
    
    # Huffman encoding demonstration
    print(f"\n" + "="*60)
    print("HUFFMAN ENCODING FEATURES")
    print("="*60)
    
    # Demonstrate Huffman features with sample data
    sample_row = X.iloc[0]
    extractor = PhishingFeatureExtractor()
    row_str = ' '.join(str(val) for val in sample_row.values)
    huffman_features = extractor.extract_huffman_features(row_str)
    
    print(f"\nSample Huffman Features:")
    for feature, value in huffman_features.items():
        print(f"  {feature}: {value:.4f}")
    
    return training_results

if __name__ == "__main__":
    results = main()