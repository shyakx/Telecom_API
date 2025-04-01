from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder, StandardScaler
import uvicorn
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1" 

# Initialize FastAPI app
app = FastAPI(
    title="Churn Prediction API (7 Features)",
    description="API to predict customer churn using 7 specific features",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define input data model for Swagger UI
class ChurnInput(BaseModel):
    account_length: int
    international_plan: str
    voice_mail_plan: str
    total_day_minutes: float
    total_eve_minutes: float
    total_night_minutes: float
    total_intl_minutes: float

    class Config:
        schema_extra = {
            "example": {
                "account_length": 128,
                "international_plan": "No",
                "voice_mail_plan": "Yes",
                "total_day_minutes": 265.1,
                "total_eve_minutes": 197.4,
                "total_night_minutes": 244.7,
                "total_intl_minutes": 10.0
            }
        }

# Define paths
DATA_PATH = os.path.join(os.getcwd(), "telecom_churn.csv")
MODEL_PATH = os.path.join(os.getcwd(), "best_model_keras.h5")

# Load the trained model
try:
    model = load_model(MODEL_PATH)
except FileNotFoundError:
    raise Exception(f"Model file not found at {MODEL_PATH}. Please run retrain_churn_model.py first.")

# Initialize preprocessors
label_encoders = {
    'International plan': LabelEncoder(),
    'Voice mail plan': LabelEncoder()
}
scaler = StandardScaler()

# Load training data to fit preprocessors
try:
    df = pd.read_csv(DATA_PATH)
except FileNotFoundError:
    raise Exception(f"Data file not found at {DATA_PATH}")

# Define features (corrected typo)
features = ['Account length', 'International plan', 'Voice mail plan', 
            'Total day minutes', 'Total eve minutes', 'Total night minutes', 
            'Total intl minutes']

# Fit preprocessors
df = df[features]  # Keep only specified features
df['International plan'] = df['International plan'].str.lower()
df['Voice mail plan'] = df['Voice mail plan'].str.lower()
label_encoders['International plan'].fit(df['International plan'])
label_encoders['Voice mail plan'].fit(df['Voice mail plan'])
X_train = df.copy()
X_train['International plan'] = label_encoders['International plan'].transform(X_train['International plan'])
X_train['Voice mail plan'] = label_encoders['Voice mail plan'].transform(X_train['Voice mail plan'])
scaler.fit(X_train)

@app.post("/predict_churn/")
async def predict_churn(input_data: ChurnInput):
    try:
        # Convert input data to dictionary
        data = input_data.dict()

        # Preprocess categorical variables
        data['international_plan'] = label_encoders['International plan'].transform([data['international_plan'].lower()])[0]
        data['voice_mail_plan'] = label_encoders['Voice mail plan'].transform([data['voice_mail_plan'].lower()])[0]  # Fixed variable name

        # Create DataFrame with correct column order
        input_df = pd.DataFrame([[
            data['account_length'],
            data['international_plan'],  # Now using the transformed numerical value
            data['voice_mail_plan'],     # Now using the transformed numerical value
            data['total_day_minutes'],
            data['total_eve_minutes'],
            data['total_night_minutes'],
            data['total_intl_minutes'],
        ]], columns=features)

        # Scale the input
        input_scaled = scaler.transform(input_df)

        # Make prediction
        prediction_proba = model.predict(input_scaled, verbose=0)[0][0]
        prediction = int(prediction_proba > 0.5)

        # Return result
        return {
            "churn_probability": float(prediction_proba),
            "churn_prediction": bool(prediction),
            "message": "True indicates customer will churn, False indicates they won't"
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing request: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

