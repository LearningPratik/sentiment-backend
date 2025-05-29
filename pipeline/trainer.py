from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report
from utils.logger import get_logger


logger = get_logger(__name__)

def build_model():
    return Pipeline([
        ('cv', CountVectorizer()),
        ('clf', LogisticRegression(solver = 'liblinear'))
    ])

def train_and_evaluate(df):
    try:
        X_train, X_test, y_train, y_test = train_test_split(df['clean_text'], df['label'], test_size = 0.2, random_state = 1)
        logger.info(f"X_train shape : {X_train.shape}, y_train: {y_train.shape}")

        model = build_model()
        model.fit(X_train, y_train)

        logger.info(f"model trained..")
        y_pred = model.predict(X_test)
        logger.info("Model training complete...")
        logger.info(f"Accuracy: {accuracy_score(y_test, y_pred)}")
        logger.info(f"Classification Report:\n{classification_report(y_test, y_pred)}")
        return model
    
    except Exception as e:
        logger.error(f"Training error: {e}")
        raise