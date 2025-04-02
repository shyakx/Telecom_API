from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report
import io
import base64
import matplotlib.pyplot as plt
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

# Define input data model for prediction endpoint
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

# Features list
features = ['Account length', 'International plan', 'Voice mail plan', 
            'Total day minutes', 'Total eve minutes', 'Total night minutes', 
            'Total intl minutes']

# Initialize model and preprocessors
model = None
label_encoders = {
    'International plan': LabelEncoder(),
    'Voice mail plan': LabelEncoder()
}
scaler = StandardScaler()

# Sample data to initialize the model (replace with your actual data if available)
sample_data = pd.DataFrame({
    'Account length': [128, 107, 137, 84],
    'International plan': ['No', 'No', 'Yes', 'No'],
    'Voice mail plan': ['Yes', 'Yes', 'No', 'No'],
    'Total day minutes': [265.1, 161.6, 243.4, 299.4],
    'Total eve minutes': [197.4, 195.5, 121.2, 61.9],
    'Total night minutes': [244.7, 254.4, 162.6, 196.9],
    'Total intl minutes': [10.0, 13.7, 12.2, 6.6],
    'Churn': [0, 0, 1, 0]
})

# Function to initialize the model with sample data
def initialize_model():
    global model, label_encoders, scaler
    df = sample_data.copy()
    
    # Encode categorical variables
    for col in ['International plan', 'Voice mail plan']:
        df[col] = df[col].str.lower()
        label_encoders[col].fit(df[col])
        df[col] = label_encoders[col].transform(df[col])

    # Define X and y
    X = df[features]
    y = df['Churn']

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Scale features
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Build and train model
    model = Sequential([
        Dense(64, activation='relu', input_shape=(len(features),)),
        Dense(32, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2, verbose=0)

# Initialize the model at startup
initialize_model()

# Prediction endpoint
@app.post("/predict_churn/")
async def predict_churn(input_data: ChurnInput):
    try:
        data = input_data.dict()
        data['international_plan'] = label_encoders['International plan'].transform([data['international_plan'].lower()])[0]
        data['voice_mail_plan'] = label_encoders['Voice mail plan'].transform([data['voice_mail_plan'].lower()])[0]

        input_df = pd.DataFrame([[
            data['account_length'],
            data['international_plan'],
            data['voice_mail_plan'],
            data['total_day_minutes'],
            data['total_eve_minutes'],
            data['total_night_minutes'],
            data['total_intl_minutes'],
        ]], columns=features)

        input_scaled = scaler.transform(input_df)
        prediction_proba = model.predict(input_scaled, verbose=0)[0][0]
        prediction = int(prediction_proba > 0.5)

        return {
            "churn_probability": float(prediction_proba),
            "churn_prediction": bool(prediction),
            "message": "True indicates customer will churn, False indicates they won't"
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing request: {str(e)}")

# Retrain endpoint (unchanged)
@app.post("/retrain_model/")
async def retrain_model(file: UploadFile):
    global model, label_encoders, scaler
    try:
        # Read uploaded CSV file in memory
        contents = await file.read()
        df = pd.read_csv(io.StringIO(contents.decode('utf-8')))
        df.columns = df.columns.str.strip()

        # Ensure Churn is integer
        df['Churn'] = df['Churn'].astype(int)

        # Keep only required features and target
        df = df[features + ['Churn']]

        # Encode categorical variables
        for col in ['International plan', 'Voice mail plan']:
            df[col] = df[col].str.lower()
            label_encoders[col].fit(df[col])
            df[col] = label_encoders[col].transform(df[col])

        # Define X and y
        X = df[features]
        y = df['Churn']

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Scale features
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # Build and train model
        model = Sequential([
            Dense(64, activation='relu', input_shape=(len(features),)),
            Dense(32, activation='relu'),
            Dense(1, activation='sigmoid')
        ])
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2, verbose=0)

        # Evaluate model
        y_pred_proba = model.predict(X_test, verbose=0)
        y_pred = (y_pred_proba > 0.5).astype(int).ravel()

        metrics = {
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "precision": float(precision_score(y_test, y_pred)),
            "recall": float(recall_score(y_test, y_pred)),
            "f1_score": float(f1_score(y_test, y_pred)),
            "roc_auc": float(roc_auc_score(y_test, y_pred)),
            "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
            "classification_report": classification_report(y_test, y_pred, output_dict=True)
        }

        # Generate plots
        fig = plt.figure(figsize=(12, 4))
        
        # Accuracy plot
        plt.subplot(1, 2, 1)
        plt.plot(history.history['accuracy'], label='Train Accuracy')
        plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
        plt.title('Model Accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend()

        # Loss plot
        plt.subplot(1, 2, 2)
        plt.plot(history.history['loss'], label='Train Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title('Model Loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend()

        # Convert plot to base64 string
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        plot_data = base64.b64encode(buf.getvalue()).decode('utf-8')
        plt.close()

        return {
            "message": "Model retrained successfully",
            "metrics": metrics,
            "plot": f"data:image/png;base64,{plot_data}"
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error retraining model: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
