#  Phishing URL Detection System

An advanced phishing detection system that combines **machine learning**, **Huffman encoding features**, and **real-time web scraping** to detect potentially malicious URLs with high accuracy. This project includes a 4-model ensemble classifier and an interactive frontend dashboard for URL analysis.

---

##  Features

-  Huffman-based compression & entropy features
-  Ensemble of Random Forest, XGBoost, SVM, and Logistic Regression
-  Feature selection using `SelectKBest` for improved performance
-  Web scraping & WHOIS for domain-based features
-  Flask API backend for analysis
-  Interactive HTML Dashboard UI for user-friendly access
-  Risk-based feature interpretation with recommendations

---

## Architecture

1. **Backend**:
   - `claudemodifiedensemble.py`: Core ML logic and Huffman-based phishing detection pipeline
   - `claudemodifiedensembletesting.py`: Feature extractor for live URL analysis
   - `server.py`: Flask REST API serving predictions at `/analyze`

2. **Frontend**:
   - `testinggui.html`: Responsive web interface with form input and visual feedback

---

##  Ensemble Models Used

- `RandomForestClassifier`
- `XGBoost` or `GradientBoostingClassifier` (fallback)
- `SVC (RBF kernel)`
- `LogisticRegression`

These models are combined using **VotingClassifier (soft voting)** weighted by cross-validation accuracy.

---

##  Setup Instructions

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Train the model(Optional)

By default, the model is loaded from optimized_phishing_detector.joblib. To retrain:
```bash
from claudemodifiedensemble import main
main()
```

### 3. Run the Backend

```bash
python server.py
```
The Flask server will start at http://localhost:5000.

### 4. Run the Frontend

Open `testinggui.html` in your browser. Enter a URL to analyze in real-time using the backend API.

![Legitimate URL](https://github.com/incognitoalpha/PhishGuard-4X-/blob/main/legitimate.png)

### Legitimate URL Result

![Feature Analysis of Legitimate URL](https://github.com/incognitoalpha/PhishGuard-4X-/blob/main/Legitimate%20feature.png)

### Feature Analysis of Legitimate URL

![Phishing URL](https://github.com/incognitoalpha/PhishGuard-4X-/blob/main/phishing%20web.png)

### Phishing URL Results

![Feature Analysis of Phishing URL](https://github.com/incognitoalpha/PhishGuard-4X-/blob/main/phishing%20feature.png)

### Feature Analysis of Phishing URL

---

## API Endpoint

### Request

```bash
{ "url": "https://example.com" }
```

### Response

```bash
{
  "url": "...",
  "prediction": "PHISHING" | "LEGITIMATE",
  "confidence": 92.5,
  "phishing_probability": 92.5,
  "legitimate_probability": 7.5,
  "is_phishing": true,
  "high_risk_features": [...],
  "features": { ... }
}
```

---

## Huffman-Based Feature Engineering

The system extracts entropy, compression ratio, and complexity metrics from feature strings using Huffman coding. These are appended to traditional phishing features to boost model robustness.

---

## Eample URLs for Testing

Try these:
- `Legit:` https://www.google.com
- `Phishing:` http://very-long-suspicious-url-with-symbol@malicious-site.com/fake-login

---

## Future Work

-`LSTM/GRU for sequence-based feature analysis`
-`Real-time browser extension integration`
-`Visual phishing detection using image hashing`
-`Development of Browser Extennsion`

---

##  License

This project is licensed under the [MIT License](https://opensource.org/licenses/MIT).

---

##  Acknowledgements

Built using:

- [Scikit-learn](https://scikit-learn.org/)
- [XGBoost](https://xgboost.ai/)
- [Flask](https://flask.palletsprojects.com/)
- [BeautifulSoup](https://www.crummy.com/software/BeautifulSoup/)
- [tldextract](https://github.com/john-kurkowski/tldextract)
