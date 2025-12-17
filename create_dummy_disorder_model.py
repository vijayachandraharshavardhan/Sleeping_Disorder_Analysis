import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
import numpy as np

# Create dummy data for disorder model (8 features)
X = np.random.rand(100, 8)  # 100 samples, 8 features
y = np.random.randint(0, 3, 100)  # 3 classes for sleep disorder

# Train dummy model
model = RandomForestClassifier()
model.fit(X, y)

# Create dummy scaler
scaler = StandardScaler()
scaler.fit(X)

# Create dummy encoders
le_disorder = LabelEncoder()
le_disorder.fit(['None', 'Insomnia', 'Sleep Apnea'])

le_gender = LabelEncoder()
le_gender.fit(['Male', 'Female'])

le_bmi = LabelEncoder()
le_bmi.fit(['Normal', 'Overweight', 'Obese', 'Underweight'])

le_bp = LabelEncoder()
le_bp.fit(['120/80', '125/80', '130/85'])

# Save model, scaler, and encoders
joblib.dump(model, 'ml/disorder_model.pkl')
joblib.dump(scaler, 'ml/disorder_scaler.pkl')
joblib.dump(le_disorder, 'ml/disorder_encoder.pkl')
joblib.dump(le_gender, 'ml/gender_encoder.pkl')
joblib.dump(le_bmi, 'ml/bmi_encoder.pkl')
joblib.dump(le_bp, 'ml/bp_encoder.pkl')

print("Dummy disorder model and related files created.")
