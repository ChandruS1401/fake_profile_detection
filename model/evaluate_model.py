"""
File: model/evaluate_model.py
Purpose: Evaluate saved model on new data
"""

import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                           f1_score, confusion_matrix, classification_report)

def evaluate_saved_model(test_data_path='dataset/test_data.csv'):
    """
    Evaluate the saved model on new data
    """
    print("Loading saved model and components...")
    
    try:
        # Load model and components
        model = joblib.load('model/fake_profile_model.joblib')
        scaler = joblib.load('model/scaler.joblib')
        feature_names = joblib.load('model/feature_names.joblib')
        
        print("Model loaded successfully!")
    except FileNotFoundError as e:
        print(f"Error loading model: {e}")
        return
    
    # Load test data
    print(f"\nLoading test data from {test_data_path}...")
    try:
        test_df = pd.read_csv(test_data_path)
        
        # Ensure test data has required features
        if 'fake' not in test_df.columns:
            print("Test data must have 'fake' column as target")
            return
        
        X_test = test_df[feature_names]
        y_test = test_df['fake']
        
        # Scale features
        X_test_scaled = scaler.transform(X_test)
        
        # Make predictions
        y_pred = model.predict(X_test_scaled)
        y_pred_proba = model.predict_proba(X_test_scaled)
        
        # Evaluate
        print("\n" + "="*50)
        print("EVALUATION ON NEW DATA")
        print("="*50)
        
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        print(f"\nPerformance Metrics:")
        print(f"Accuracy:  {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall:    {recall:.4f}")
        print(f"F1-Score:  {f1:.4f}")
        
        print("\nConfusion Matrix:")
        print(confusion_matrix(y_test, y_pred))
        
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=['Real', 'Fake']))
        
        # Show some predictions with probabilities
        print("\nSample Predictions:")
        for i in range(min(5, len(y_test))):
            actual = "Fake" if y_test.iloc[i] == 1 else "Real"
            predicted = "Fake" if y_pred[i] == 1 else "Real"
            fake_prob = y_pred_proba[i][1] * 100
            
            print(f"Sample {i+1}: Actual={actual}, Predicted={predicted}, "
                  f"Fake Probability={fake_prob:.1f}%")
        
    except Exception as e:
        print(f"Error during evaluation: {e}")

if __name__ == "__main__":
    evaluate_saved_model()