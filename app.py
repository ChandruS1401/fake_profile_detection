"""
File: app.py (FIXED VERSION)
Purpose: Main Flask application with supervised learning integration
"""

from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np
import pandas as pd
import os

app = Flask(__name__)

# Load supervised learning model and components
def load_model():
    """Load the trained supervised learning model"""
    try:
        model = joblib.load('model/fake_profile_model.joblib')
        feature_names = joblib.load('model/feature_names.joblib')
        print("✅ Supervised Learning Model loaded successfully!")
        print(f"📊 Model Type: {type(model).__name__}")
        print(f"📋 Features: {feature_names}")
        
        # No scaler needed for our simple model
        return model, feature_names
    except FileNotFoundError as e:
        print(f"❌ Error loading model: {e}")
        print("💡 Please run: python model/train_model.py")
        return None, None

# Initialize model
model, feature_names = load_model()

def predict_profile(features):
    """
    Make prediction using supervised learning model
    """
    if model is None:
        return None, None, None
    
    try:
        # Create DataFrame with correct feature order
        input_df = pd.DataFrame([features], columns=feature_names)
        
        # No scaling needed - use raw features
        input_data = input_df.values
        
        # Make prediction
        prediction = model.predict(input_data)[0]
        probabilities = model.predict_proba(input_data)[0]
        
        return prediction, probabilities, input_df.iloc[0].to_dict()
    
    except Exception as e:
        print(f"Prediction error: {e}")
        return None, None, None

def analyze_risk_factors(features):
    """
    Analyze risk factors based on heuristics
    """
    risk_factors = []
    recommendations = []
    
    followers = features.get('followers', 0)
    following = features.get('following', 0)
    posts = features.get('posts', 0)
    account_age = features.get('account_age', 0)
    profile_complete = features.get('profile_complete', 0)
    
    # Risk analysis
    if following > followers * 3:
        risk_factors.append("High following-to-follower ratio (suspicious)")
        recommendations.append("Consider reducing following count")
    
    if posts < 10 and account_age > 30:
        risk_factors.append("Very low post count for account age")
        recommendations.append("Increase posting activity")
    
    if account_age < 7:
        risk_factors.append("Very new account")
        recommendations.append("Wait for account to mature")
    
    if profile_complete == 0:
        risk_factors.append("Incomplete profile")
        recommendations.append("Complete profile information")
    
    if followers > 10000 and posts < 50:
        risk_factors.append("High follower count with low activity")
    
    return risk_factors, recommendations

@app.route('/')
def home():
    """Render the main input form"""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handle prediction requests"""
    if model is None:
        return '''
        <h2>❌ Model Not Loaded</h2>
        <p>Please train the model first:</p>
        <code>python model/train_model.py</code>
        <br><br>
        <a href="/">Go Back</a>
        '''
    
    try:
        # Get form data
        followers = float(request.form['followers'])
        following = float(request.form['following'])
        posts = float(request.form['posts'])
        account_age = float(request.form['account_age'])
        profile_complete = int(request.form['profile_complete'])
        private = int(request.form['private'])
        
        # Prepare features
        features = {
            'followers': followers,
            'following': following,
            'posts': posts,
            'account_age': account_age,
            'profile_complete': profile_complete,
            'private': private
        }
        
        # Make prediction using supervised learning
        prediction, probabilities, processed_features = predict_profile(features)
        
        if prediction is None:
            return f'''
            <h2>❌ Prediction Failed</h2>
            <p>Error processing input data</p>
            <a href="/">Go Back</a>
            '''
        
        # Determine result
        result = "FAKE" if prediction == 1 else "REAL"
        confidence = max(probabilities) * 100
        fake_probability = probabilities[1] * 100
        
        # Analyze risk factors
        risk_factors, recommendations = analyze_risk_factors(features)
        
        # Get feature importance if available
        feature_importance = {}
        if hasattr(model, 'feature_importances_'):
            feature_importance = dict(zip(feature_names, model.feature_importances_))
        
        # Simple HTML response (since templates might not exist)
        risk_html = ""
        if risk_factors:
            risk_html = "<h3>⚠️ Risk Factors:</h3><ul>"
            for risk in risk_factors:
                risk_html += f"<li>{risk}</li>"
            risk_html += "</ul>"
        
        rec_html = ""
        if recommendations:
            rec_html = "<h3>💡 Recommendations:</h3><ul>"
            for rec in recommendations:
                rec_html += f"<li>{rec}</li>"
            rec_html += "</ul>"
        
        return f'''
        <!DOCTYPE html>
        <html>
        <head>
            <title>Detection Result</title>
            <style>
                body {{ font-family: Arial; padding: 20px; max-width: 800px; margin: 0 auto; }}
                .result {{ font-size: 24px; font-weight: bold; padding: 20px; text-align: center; border-radius: 10px; margin: 20px 0; }}
                .real {{ background: #d4edda; color: #155724; border: 2px solid #c3e6cb; }}
                .fake {{ background: #f8d7da; color: #721c24; border: 2px solid #f5c6cb; }}
                .data-table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
                .data-table td {{ padding: 10px; border-bottom: 1px solid #ddd; }}
                a {{ color: #007bff; text-decoration: none; }}
            </style>
        </head>
        <body>
            <h1>🔍 Detection Result</h1>
            
            <div class="result {result.lower()}">
                {result} PROFILE
            </div>
            
            <p><strong>Confidence:</strong> {confidence:.1f}%</p>
            <p><strong>Fake Probability:</strong> {fake_probability:.1f}%</p>
            
            <h3>📊 Profile Details:</h3>
            <table class="data-table">
                <tr><td><strong>Followers:</strong></td><td>{followers}</td></tr>
                <tr><td><strong>Following:</strong></td><td>{following}</td></tr>
                <tr><td><strong>Posts:</strong></td><td>{posts}</td></tr>
                <tr><td><strong>Account Age:</strong></td><td>{account_age} days</td></tr>
                <tr><td><strong>Profile Complete:</strong></td><td>{'Yes' if profile_complete==1 else 'No'}</td></tr>
                <tr><td><strong>Private:</strong></td><td>{'Yes' if private==1 else 'No'}</td></tr>
            </table>
            
            {risk_html}
            {rec_html}
            
            <br>
            <a href="/">← Check Another Profile</a>
        </body>
        </html>
        '''
    
    except Exception as e:
        return f'''
        <h2>❌ Error</h2>
        <p>{str(e)}</p>
        <a href="/">Go Back</a>
        '''

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """API endpoint for programmatic access"""
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    try:
        data = request.get_json()
        
        # Check if we have feature names
        if feature_names is None:
            return jsonify({'error': 'Feature names not loaded'}), 500
        
        # Validate required fields
        for field in feature_names:
            if field not in data:
                return jsonify({'error': f'Missing field: {field}'}), 400
        
        # Prepare features
        features = {field: float(data[field]) for field in feature_names}
        
        # Make prediction
        prediction, probabilities, _ = predict_profile(features)
        
        if prediction is None:
            return jsonify({'error': 'Prediction failed'}), 500
        
        return jsonify({
            'prediction': int(prediction),
            'label': 'fake' if prediction == 1 else 'real',
            'confidence': round(max(probabilities) * 100, 2),
            'probabilities': {
                'real': round(probabilities[0] * 100, 2),
                'fake': round(probabilities[1] * 100, 2)
            },
            'features': features
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/model_info', methods=['GET'])
def model_info():
    """Get model information"""
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    return jsonify({
        'model_type': 'RandomForestClassifier',
        'supervised_learning': True,
        'n_features': len(feature_names) if feature_names else 0,
        'features': feature_names if feature_names else [],
        'n_estimators': model.n_estimators if hasattr(model, 'n_estimators') else 'Unknown'
    })

@app.route('/test')
def test():
    """Test endpoint"""
    return '''
    <h1>✅ Fake Profile Detection System</h1>
    <p>System is working!</p>
    <p>Go to <a href="/">Home Page</a> to test the detector.</p>
    '''

if __name__ == '__main__':
    # Create necessary directories
    for dir_name in ['model', 'templates', 'static', 'dataset', 'utils']:
        os.makedirs(dir_name, exist_ok=True)
    
    # Run the Flask app
    print("=" * 60)
    print("🚀 FAKE PROFILE DETECTION SYSTEM")
    print("=" * 60)
    print("🌐 Open: http://localhost:5000")
    print("🌐 Test: http://localhost:5000/test")
    print("=" * 60)
    app.run(debug=True, host='0.0.0.0', port=5000)      