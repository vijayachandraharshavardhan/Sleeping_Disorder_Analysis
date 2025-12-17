import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import numpy as np

# Create dummy data
X = np.random.rand(100, 1)  # 100 samples, 1 feature (BPM)
y = np.random.randint(0, 2, 100)  # Binary classification

# Train dummy model
model = RandomForestClassifier()
model.fit(X, y)

# Create dummy scaler
scaler = StandardScaler()
scaler.fit(X)

# Save model and scaler
joblib.dump(model, 'ml/model.pkl')
joblib.dump(scaler, 'ml/scaler.pkl')

print("Dummy model and scaler created.")
