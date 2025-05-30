from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from pipeline.predictor import load_model, predict_sentiment
from pipeline.preprocessor import clean_text
from config import MODEL_PATH
from utils.logger import get_logger
from fastapi.middleware.cors import CORSMiddleware
import joblib


logger = get_logger(__name__)

# Initialize fastapi app
app = FastAPI(title = "Sentiment Analyzer API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://sentiment-analyz.netlify.app"],  # In production, replace "*" with your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# load model at startup
# try:
#     model = load_model(MODEL_PATH)
# except Exception as e:
#     logger.critical(f"Failed to load the model : {e}")
#     model = None

model = joblib.load(MODEL_PATH)

# Request body
class SentimentRequest(BaseModel):
    text: str

# Response body
class SentimentResponse(BaseModel):
    sentiment: str

@app.post("/predict", response_model = SentimentResponse)
def predict(request: SentimentRequest):

    if not model:
        raise HTTPException(status_code = 500, detail = "Model not available")
    if not request.text:
        raise HTTPException(status_code = 400, detail = "Text input is required")
    
    # sentiment = predict_sentiment(model, request.text)
    cleaned = clean_text(text)
    sentiment = model.predict(cleaned)[0]
    
    if sentiment is None:
        raise HTTPException(status_code=500, detail="Prediction failed")

    return SentimentResponse(sentiment=sentiment)
