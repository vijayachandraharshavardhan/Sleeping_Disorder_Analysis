import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib
import requests
import zipfile
import io

# Download dataset from Kaggle
url = 'https://www.kaggle.com/api/v1/datasets/download/uom190346a/sleep-health-and-lifestyle-dataset'
response = requests.get(url)
z = zipfile.ZipFile(io.BytesIO(response.content))
z.extractall('.')

# Load dataset
df = pd.read_csv('Sleep_health_and_lifestyle_dataset.csv')

# Preprocess
le_gender = LabelEncoder()
le_bmi = LabelEncoder()
le_bp = LabelEncoder()
le_occupation = LabelEncoder()
le_disorder = LabelEncoder()

df['Gender'] = le_gender.fit_transform(df['Gender'])
df['BMI Category'] = le_bmi.fit_transform(df['BMI Category'])
df['Blood Pressure'] = le_bp.fit_transform(df['Blood Pressure'])
df['Occupation'] = le_occupation.fit_transform(df['Occupation'])
df['Sleep Disorder'] = le_disorder.fit_transform(df['Sleep Disorder'])

# Features: Heart Rate, Age, Gender, BMI, etc. (add snore later)
features = ['Heart Rate', 'Age', 'Gender', 'BMI Category', 'Sleep Duration', 'Quality of Sleep', 'Stress Level', 'Physical Activity Level']
X = df[features]
y = df['Sleep Disorder']

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Scaler
scaler = StandardScaler()
scaler.fit(X_train)

# Save
joblib.dump(model, 'ml/disorder_model.pkl')
joblib.dump(scaler, 'ml/disorder_scaler.pkl')
joblib.dump(le_disorder, 'ml/disorder_encoder.pkl')
joblib.dump(le_gender, 'ml/gender_encoder.pkl')
joblib.dump(le_bmi, 'ml/bmi_encoder.pkl')
joblib.dump(le_bp, 'ml/bp_encoder.pkl')

print("Model trained and saved.")
