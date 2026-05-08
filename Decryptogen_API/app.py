from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
import uvicorn
from fastapi.middleware.cors import CORSMiddleware

# FastAPI app
app = FastAPI(
    title="Personality Predictor API",
    description="API to predict whether a person is an Introvert or Extrovert based on behaviors.",
    version="1.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Allows all websites to connect
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return {"message": "Welcome to the Personality Predictor API. Send a POST request to /predict."}



# Loading the trained Logistic Regression Pipeline
# (This includes both the StandardScaler and the Model)
import os
import joblib

MODEL_PATH = os.path.join(os.path.dirname(__file__), "personality_pipeline.pkl")
model_pipeline = joblib.load(MODEL_PATH)



# Define the Input Data Schema using Pydantic
class PersonalityFeatures(BaseModel):
    Time_spent_Alone: float
    Stage_fear: str
    Social_event_attendance: float
    Going_outside: float
    Drained_after_socializing: str
    Friends_circle_size: float
    Post_frequency: float



# Create the POST endpoint
@app.post("/predict")
def predict_personality(data: PersonalityFeatures):
    # Convert incoming JSON data into a dictionary
    input_data = data.model_dump()
    
    # Preprocess the "Yes"/"No" text into 1/0
    binary_mapping = {'Yes': 1, 'No': 0}
    stage_fear_num = binary_mapping.get(input_data['Stage_fear'], 0)
    drained_num = binary_mapping.get(input_data['Drained_after_socializing'], 0)
    
    # Format the data into a Pandas DataFrame 
    # (The pipeline requires a DataFrame so it knows the column names for scaling)
    features_df = pd.DataFrame([{
        'Time_spent_Alone': input_data['Time_spent_Alone'],
        'Stage_fear': stage_fear_num,
        'Social_event_attendance': input_data['Social_event_attendance'],
        'Going_outside': input_data['Going_outside'],
        'Drained_after_socializing': drained_num,
        'Friends_circle_size': input_data['Friends_circle_size'],
        'Post_frequency': input_data['Post_frequency']
    }])
    
    # Make the prediction (The pipeline automatically scales the data first!)
    prediction_num = model_pipeline.predict(features_df)[0]
    
    # Convert the 1/0 back to human-readable text
    prediction_text = "Extrovert" if prediction_num == 1 else "Introvert"
    
    # Return the JSON response
    return {
        "prediction": prediction_text,
        "model_used": "Logistic Regression (Standardized)"
    }



if __name__ == "__main__":
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)