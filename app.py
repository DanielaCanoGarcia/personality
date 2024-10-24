from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import pandas as pd
from joblib import load
import pathlib
from fastapi.middleware.cors import CORSMiddleware

origins = ["*"]
app = FastAPI(title='Personality Prediction')
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

model = load(pathlib.Path('model/personality-pipeline-v1.joblib'))

class InputData(BaseModel):
    Age: int
    Gender: str
    Education: int  
    IntroversionScore: float
    SensingScore: float
    ThinkingScore: float
    JudgingScore: float
    Interest: str

class OutputData(BaseModel):
    score: float

@app.post('/score', response_model=OutputData)
def score(data: InputData):
    try:
        model_input = pd.DataFrame([data.dict()])
        result = model.predict_proba(model_input)[:, -1][0]
        return {'score': result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
