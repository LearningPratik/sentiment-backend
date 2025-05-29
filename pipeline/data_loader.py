import pandas as pd
from pipeline.preprocessor import clean_text
from utils.logger import get_logger

logger = get_logger(__name__)

def load_dataset(filepath):

    try:
        df = pd.read_csv(filepath)
        logger.info(f"Shape of data : {df.shape}")
        logger.info(f"columns in data : {df.columns}")
        df.dropna(subset = ['review', 'label'], inplace=True)
        df['clean_text'] = df['review'].apply(clean_text)
        return df[['clean_text', 'label']]
    
    except Exception as e:
        logger.error(f"Error loading dataset: {e}")
        raise