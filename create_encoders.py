import pandas as pd
from sklearn.preprocessing import LabelEncoder
import joblib

# Load dataset
df = pd.read_csv('Sleep_health_and_lifestyle_dataset.csv')

# Fit encoders with actual data
le_gender = LabelEncoder()
le_gender.fit(df['Gender'].unique())

le_bmi = LabelEncoder()
le_bmi.fit(df['BMI Category'].unique())

le_bp = LabelEncoder()
le_bp.fit(df['Blood Pressure'].unique())

# Save encoders
joblib.dump(le_gender, 'ml/gender_encoder.pkl')
joblib.dump(le_bmi, 'ml/bmi_encoder.pkl')
joblib.dump(le_bp, 'ml/bp_encoder.pkl')

print("Encoders fitted with dataset and saved.")

