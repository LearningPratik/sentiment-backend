import joblib
from pipeline.preprocessor import clean_text
from utils.logger import get_logger

logger = get_logger(__name__)

def save_model(model, filepath):
    try:
        with open(filepath, 'wb') as f:
            joblib.dump(model, f)
        logger.info(f"Model saved to {filepath}")

    except Exception as e:
        logger.error(f"Error saving model: {e}")
        raise

def load_model(filepath):
    try:
        with open(filepath, 'rb') as f:
            logger.info(f"Model loaded from {filepath}")
            return joblib.load(f)
    except Exception as e:
        logger.error(f"Error loading the model: {e}")
        raise

def predict_sentiment(model, text):
    try:
        cleaned_text = clean_text(text)
        return model.predict([cleaned_text])[0]
    except Exception as e:
        logger.error(f"Prediction error : {e}")
        return None