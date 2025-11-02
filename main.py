from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
from pydantic import BaseModel
import numpy as np
import pandas as pd
import pickle
import tensorflow as tf
from tensorflow.keras.models import load_model
import os
from typing import Dict, Any

# Initialize FastAPI app
app = FastAPI(title="Fraud Detection API")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create static directory if it doesn't exist
static_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
os.makedirs(static_dir, exist_ok=True)

# Mount static files (for serving the HTML)
app.mount("/static", StaticFiles(directory=static_dir), name="static")

# Get the base directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
print(f"Current working directory: {os.getcwd()}")
print(f"Script directory: {BASE_DIR}")

# Define file paths
MODEL_PATH = os.path.join(BASE_DIR, "professional_fraud_model.h5")
PREPROCESSOR_PATH = os.path.join(BASE_DIR, "professional_preprocessor.pkl")

print(f"Looking for model at: {MODEL_PATH}")
print(f"Looking for preprocessor at: {PREPROCESSOR_PATH}")

# Check if model and preprocessor exist
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found at: {MODEL_PATH}")
if not os.path.exists(PREPROCESSOR_PATH):
    raise FileNotFoundError(f"Preprocessor file not found at: {PREPROCESSOR_PATH}")

print("Both files found. Loading model and preprocessor...")

# Initialize variables at module level
preprocessor = None
model = None

# Load the preprocessor and model
try:
    # Try loading the preprocessor with joblib (better for sklearn objects)
    import joblib
    
    try:
        preprocessor = joblib.load(PREPROCESSOR_PATH)
        print("Successfully loaded preprocessor with joblib")
    except Exception as e_joblib:
        print(f"Joblib loading failed: {str(e_joblib)}")
        
        # Fallback to pickle
        import pickle
        
        # Method 1: Try with default protocol
        try:
            with open(PREPROCESSOR_PATH, 'rb') as f:
                preprocessor = pickle.load(f)
            print("Successfully loaded preprocessor with default protocol")
        except Exception as e1:
            print(f"Default protocol failed: {str(e1)}")
            
            # Method 2: Try with latin1 encoding
            try:
                with open(PREPROCESSOR_PATH, 'rb') as f:
                    preprocessor = pickle.load(f, encoding='latin1')
                print("Successfully loaded preprocessor with latin1 encoding")
            except Exception as e2:
                print(f"Latin1 encoding failed: {str(e2)}")
                raise Exception(f"All preprocessor loading methods failed. Last error: {str(e2)}")
    
    if preprocessor is None:
        raise Exception("Preprocessor could not be loaded")
    
    # Load the model
    model = load_model(MODEL_PATH)
    print("Successfully loaded model")
    
except Exception as e:
    print(f"Error loading model or preprocessor: {str(e)}")
    # For now, let's create a dummy preprocessor to test the API
    print("Creating dummy preprocessor for testing...")
    from sklearn.preprocessing import StandardScaler, OneHotEncoder
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.impute import SimpleImputer
    
    # Create a simple preprocessor that matches expected structure
    numeric_features = ['age_of_driver', 'marital_status', 'annual_income', 'high_education_ind', 
                       'address_change_ind', 'age_of_vehicle', 'vehicle_price', 'vehicle_weight',
                       'safety_rating', 'past_num_of_claims', 'claim_est_payout', 'witness_present_ind',
                       'policy_report_filed_ind']
    categorical_features = ['gender', 'living_status', 'vehicle_category', 'vehicle_color',
                           'claim_day_of_week', 'accident_site', 'channel', 'liab_prct']
    
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ]
    )
    
    # Fit the preprocessor with sample data
    print("Fitting dummy preprocessor with sample data...")
    sample_data = pd.DataFrame([{
        'age_of_driver': 30,
        'gender': 'M',
        'marital_status': 2.0,
        'annual_income': 50000,
        'high_education_ind': 1,
        'address_change_ind': 0,
        'living_status': 'Own',
        'age_of_vehicle': 5,
        'vehicle_price': 20000,
        'vehicle_weight': 3000,
        'vehicle_category': 'Sedan',
        'vehicle_color': 'Black',
        'safety_rating': 4,
        'past_num_of_claims': 2,
        'claim_est_payout': 5000,
        'claim_day_of_week': 'Monday',
        'accident_site': 'Highway',
        'witness_present_ind': 1.0,
        'policy_report_filed_ind': 1,
        'channel': 'Phone',
        'liab_prct': '20-50'
    }])
    preprocessor.fit(sample_data)
    print("Dummy preprocessor fitted successfully")
    
    # Load the model
    model = load_model(MODEL_PATH)
    print("Successfully loaded model with dummy preprocessor")

# Verify preprocessor and model are loaded
if preprocessor is None or model is None:
    raise RuntimeError("Failed to initialize preprocessor or model")

print("\n" + "="*50)
print("Server initialization complete!")
print(f"Preprocessor type: {type(preprocessor)}")
print(f"Model type: {type(model)}")
print("="*50 + "\n")

# Define request/response models
class TransactionData(BaseModel):
    age_of_driver: int
    safty_rating: int  # Note: 'safty' not 'safety' - matches training data
    annual_income: int
    vehicle_price: int
    age_of_vehicle: int
    past_num_of_claims: int
    claim_est_payout: int
    vehicle_weight: int
    marital_status: str  # String for categorical
    high_education_ind: str  # String for categorical
    address_change_ind: str  # String for categorical
    witness_present_ind: str  # String for categorical
    liab_prct: str
    policy_report_filed_ind: str  # String for categorical
    gender: str
    living_status: str
    claim_day_of_week: str
    accident_site: str
    channel: str
    vehicle_category: str
    vehicle_color: str

class PredictionResult(BaseModel):
    is_fraud: bool
    confidence: float
    message: str

# API endpoints
@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    # Serve the index.html file
    return FileResponse(os.path.join(BASE_DIR, 'index.html'), media_type='text/html')

@app.post("/predict", response_model=PredictionResult)
async def predict_fraud(transaction: TransactionData):
    try:
        # Convert input to DataFrame with the EXACT feature names from training
        input_data = pd.DataFrame([{
            'age_of_driver': transaction.age_of_driver,
            'safty_rating': transaction.safty_rating,  # Note: 'safty' not 'safety'
            'annual_income': transaction.annual_income,
            'vehicle_price': transaction.vehicle_price,
            'age_of_vehicle': transaction.age_of_vehicle,
            'past_num_of_claims': transaction.past_num_of_claims,
            'claim_est_payout': transaction.claim_est_payout,
            'vehicle_weight': transaction.vehicle_weight,
            'marital_status': transaction.marital_status,
            'high_education_ind': transaction.high_education_ind,
            'address_change_ind': transaction.address_change_ind,
            'witness_present_ind': transaction.witness_present_ind,
            'liab_prct': transaction.liab_prct,
            'policy_report_filed_ind': transaction.policy_report_filed_ind,
            'gender': transaction.gender,
            'living_status': transaction.living_status,
            'claim_day_of_week': transaction.claim_day_of_week,
            'accident_site': transaction.accident_site,
            'channel': transaction.channel,
            'vehicle_category': transaction.vehicle_category,
            'vehicle_color': transaction.vehicle_color
        }])

        # Preprocess the input data
        try:
            # Transform the input data using the preprocessor
            processed_data = preprocessor.transform(input_data)
            
            # Make prediction
            prediction = model.predict(processed_data)
            
            # Get the probability of fraud (assuming binary classification)
            fraud_probability = float(prediction[0][0]) if isinstance(prediction[0], np.ndarray) else float(prediction[0])
            
            # Determine if it's fraud based on threshold (0.5 is a common default)
            is_fraud = fraud_probability >= 0.5
            
            # Format the confidence as a percentage
            confidence = fraud_probability if is_fraud else (1 - fraud_probability)
            
            return {
                "is_fraud": bool(is_fraud),
                "confidence": float(confidence),
                "message": "Potential fraud detected" if is_fraud else "Transaction appears legitimate"
            }
            
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Error processing input: {str(e)}")
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

# Serve the index.html file
@app.get("/index.html")
async def serve_index():
    with open("index.html", "r") as f:
        return Response(content=f.read(), media_type="text/html")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
