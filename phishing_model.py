from flask import Flask, request, jsonify, render_template_string
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import os

app = Flask(__name__)

# Load and train model on startup
print("Loading phishing detection model...")

# Load dataset
data = pd.read_csv("dataset/phishing.csv")

# Separate features and target
X = data.drop("Result", axis=1)
y = data["Result"]

# Convert labels: phishing (-1) → 1, legitimate (1) → 0
y = y.replace({-1: 1, 1: 0})

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train supervised model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"Model trained! Accuracy: {accuracy * 100:.2f}%")

# Get feature names for API
feature_names = list(X.columns)

# HTML Template
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Phishing Detection API</title>
    <style>
        body { font-family: Arial; max-width: 900px; margin: 50px auto; padding: 20px; background: #f5f5f5; }
        h1 { color: #d32f2f; }
        .info { background: white; padding: 20px; border-radius: 8px; margin: 20px 0; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        .status { color: #4caf50; font-weight: bold; }
        code { background: #e0e0e0; padding: 2px 6px; border-radius: 3px; }
        pre { background: #263238; color: #aed581; padding: 15px; border-radius: 5px; overflow-x: auto; }
        ul { line-height: 1.8; }
        .features { columns: 3; font-size: 12px; }
    </style>
</head>
<body>
    <h1>🔒 Phishing Detection API</h1>
    
    <div class="info">
        <p><strong>Model Accuracy:</strong> {{ accuracy }}%</p>
        <p><strong>Status:</strong> <span class="status">Running ✅</span></p>
        <p><strong>Model Type:</strong> Logistic Regression</p>
        <p><strong>Features:</strong> {{ num_features }} URL/Website characteristics</p>
    </div>
    
    <h2>📡 API Endpoints:</h2>
    <ul>
        <li><code>GET /</code> - This page</li>
        <li><code>GET /health</code> - Health check</li>
        <li><code>POST /predict</code> - Detect phishing</li>
        <li><code>GET /features</code> - List all required features</li>
    </ul>
    
    <h2>📝 Example Request:</h2>
    <pre>
POST /predict
Content-Type: application/json

{
  "having_IP_Address": -1,
  "URL_Length": 1,
  "Shortining_Service": 1,
  "having_At_Symbol": 1,
  "double_slash_redirecting": -1,
  "Prefix_Suffix": -1,
  "having_Sub_Domain": -1,
  "SSLfinal_State": -1,
  "Domain_registeration_length": -1,
  "Favicon": 1,
  "port": 1,
  "HTTPS_token": -1,
  "Request_URL": 1,
  "URL_of_Anchor": -1,
  "Links_in_tags": 1,
  "SFH": -1,
  "Submitting_to_email": -1,
  "Abnormal_URL": -1,
  "Redirect": 0,
  "on_mouseover": 1,
  "RightClick": 1,
  "popUpWidnow": 1,
  "Iframe": 1,
  "age_of_domain": -1,
  "DNSRecord": -1,
  "web_traffic": -1,
  "Page_Rank": -1,
  "Google_Index": 1,
  "Links_pointing_to_page": 1,
  "Statistical_report": -1
}
    </pre>
    
    <h2>✅ Response Example:</h2>
    <pre>
{
  "prediction": "Phishing",
  "confidence": 87.5,
  "risk_level": "High",
  "is_safe": false
}
    </pre>
    
    <h2>🔧 Required Features ({{ num_features }}):</h2>
    <div class="features">
        <ul>
            {% for feature in features %}
            <li>{{ feature }}</li>
            {% endfor %}
        </ul>
    </div>
    
    <p style="margin-top: 30px; color: #666; font-size: 14px;">
        💡 <strong>Note:</strong> Feature values are typically -1, 0, or 1 based on URL characteristics.
    </p>
</body>
</html>
"""

@app.route('/')
def home():
    return render_template_string(
        HTML_TEMPLATE, 
        accuracy=round(accuracy * 100, 2),
        num_features=len(feature_names),
        features=feature_names
    )

@app.route('/health')
def health():
    return jsonify({
        'status': 'healthy',
        'model': 'Logistic Regression',
        'accuracy': round(accuracy * 100, 2),
        'features_count': len(feature_names)
    })

@app.route('/features')
def get_features():
    return jsonify({
        'features': feature_names,
        'count': len(feature_names),
        'description': 'All features must be provided as -1, 0, or 1'
    })

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get JSON data
        input_data = request.json
        
        if not input_data:
            return jsonify({'error': 'No data provided'}), 400
        
        # Check if all features are present
        missing_features = [f for f in feature_names if f not in input_data]
        if missing_features:
            return jsonify({
                'error': 'Missing features',
                'missing': missing_features,
                'required_features': feature_names
            }), 400
        
        # Create DataFrame in correct order
        input_df = pd.DataFrame([input_data])[feature_names]
        
        # Make prediction
        prediction = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0]
        
        # Interpret result
        is_phishing = bool(prediction == 1)
        result_text = "Phishing" if is_phishing else "Legitimate"
        risk_level = "High" if is_phishing else "Low"
        confidence = float(max(probability) * 100)
        
        # Return result
        return jsonify({
            'prediction': result_text,
            'is_phishing': is_phishing,
            'is_safe': not is_phishing,
            'confidence': round(confidence, 2),
            'risk_level': risk_level,
            'probabilities': {
                'legitimate': round(float(probability[0]) * 100, 2),
                'phishing': round(float(probability[1]) * 100, 2)
            }
        })
        
    except Exception as e:
        return jsonify({
            'error': 'Prediction failed',
            'details': str(e)
        }), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
