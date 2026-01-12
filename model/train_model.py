print("=" * 60)
print("🤖 FAKE PROFILE DETECTOR - SIMPLE TRAINING")
print("=" * 60)

print("\n📦 Checking required packages...")

# Basic imports (no matplotlib)
import sys

try:
    import pandas as pd
    print("✅ pandas - OK")
except ImportError:
    print("❌ pandas missing")
    print("💡 Install: pip install pandas")
    sys.exit(1)

try:
    import numpy as np
    print("✅ numpy - OK")
except ImportError:
    print("❌ numpy missing")
    print("💡 Install: pip install numpy")
    sys.exit(1)

try:
    from sklearn.ensemble import RandomForestClassifier
    print("✅ scikit-learn - OK")
except ImportError:
    print("❌ scikit-learn missing")
    print("💡 Install: pip install scikit-learn")
    sys.exit(1)

try:
    import joblib
    print("✅ joblib - OK")
except ImportError:
    print("❌ joblib missing")
    print("💡 Install: pip install joblib")
    sys.exit(1)

print("\n📊 Creating training dataset...")

# Simple training data
data = [
    # [followers, following, posts, account_age, profile_complete, private, fake]
    [1500, 200, 45, 120, 1, 0, 0],    # Real - normal user
    [50, 1500, 2, 10, 0, 1, 1],       # Fake - follows too many
    [2000, 500, 120, 365, 1, 0, 0],   # Real - established user
    [30, 800, 1, 5, 0, 1, 1],         # Fake - new, incomplete
    [800, 300, 80, 200, 1, 0, 0],     # Real - balanced
    [100, 800, 3, 20, 0, 1, 1]        # Fake - suspicious
]

# Convert to DataFrame
df = pd.DataFrame(data, columns=[
    'followers', 'following', 'posts', 
    'account_age', 'profile_complete', 'private', 'fake'
])

print(f"✅ Created {len(df)} training samples")
print("\nDataset preview:")
print(df.head())

print("\n🎯 Training Random Forest model...")

# Separate features and target
X = df[['followers', 'following', 'posts', 'account_age', 'profile_complete', 'private']]
y = df['fake']

print(f"Features shape: {X.shape}")
print(f"Target shape: {y.shape}")

# Create and train model
model = RandomForestClassifier(
    n_estimators=50,
    max_depth=5,
    random_state=42,
    class_weight='balanced'
)

model.fit(X, y)

# Calculate accuracy
train_accuracy = model.score(X, y)
print(f"✅ Training accuracy: {train_accuracy:.1%}")

print("\n💾 Saving model and files...")

# Save model
joblib.dump(model, 'fake_profile_model.joblib')

# Save feature names
feature_names = list(X.columns)
joblib.dump(feature_names, 'feature_names.joblib')

# Save dataset
df.to_csv('../dataset/training_data.csv', index=False)

print("✅ Model saved: fake_profile_model.joblib")
print("✅ Features saved: feature_names.joblib")
print("✅ Dataset saved: ../dataset/training_data.csv")

print("\n🧪 Running test predictions...")

# Test cases
test_cases = [
    {"name": "Normal User", "data": [1500, 200, 45, 120, 1, 0], "expected": "REAL"},
    {"name": "Fake Account", "data": [50, 1500, 2, 10, 0, 1], "expected": "FAKE"},
    {"name": "Suspicious", "data": [100, 500, 3, 5, 0, 1], "expected": "FAKE"},
    {"name": "Real Business", "data": [5000, 800, 300, 730, 1, 0], "expected": "REAL"}
]

print("\nTest Results:")
print("-" * 40)

for test in test_cases:
    prediction = model.predict([test['data']])[0]
    result = "FAKE" if prediction == 1 else "REAL"
    status = "✅" if result == test['expected'] else "⚠️"
    
    print(f"{status} {test['name']}: {result}")
    print(f"   Followers: {test['data'][0]}, Following: {test['data'][1]}")

print("\n" + "=" * 60)
print("🎉 TRAINING COMPLETED SUCCESSFULLY!")
print("=" * 60)
print("\n📌 Next steps:")
print("1. Go back to project folder: cd ..")
print("2. Run the web app: python app.py")
print("3. Open browser: http://localhost:5000")