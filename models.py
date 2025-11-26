import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from transformers import pipeline
import joblib


# -------------------------------------------------------------------------
# TECHNICAL MODEL (LSTM)
# -------------------------------------------------------------------------
class TechnicalModel:
    def __init__(self, sequence_length=60):
        self.sequence_length = sequence_length
        self.model = self.build_lstm_model()
        self.scaler = StandardScaler()

    def build_lstm_model(self):
        """
        Build LSTM model for technical analysis
        """
        model = Sequential([
            LSTM(50, return_sequences=True, input_shape=(self.sequence_length, 5)),
            Dropout(0.2),
            LSTM(50, return_sequences=False),
            Dropout(0.2),
            Dense(25),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='mean_squared_error')
        return model

    def prepare_data(self, df):
        """
        Prepare input sequences and labels for LSTM
        """
        features = ['Close', 'MA20', 'RSI', 'MACD', 'BB_Middle']
        data = df[features].dropna()
        scaled_data = self.scaler.fit_transform(data)

        X, y = [], []
        for i in range(self.sequence_length, len(scaled_data)):
            X.append(scaled_data[i - self.sequence_length:i])
            y.append(scaled_data[i, 0])  # Predict Close price

        return np.array(X), np.array(y)

    def train(self, X, y, epochs=50, batch_size=32):
        """
        Train LSTM model
        """
        history = self.model.fit(
            X, y,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.2,
            verbose=1
        )
        return history

    def predict(self, X):
        """
        Predict using trained LSTM
        """
        return self.model.predict(X)


# -------------------------------------------------------------------------
# SENTIMENT MODEL (DISTILBERT)
# -------------------------------------------------------------------------
class SentimentModel:
    def __init__(self):
        self.model = pipeline(
            "sentiment-analysis",
            model="distilbert-base-uncased-finetuned-sst-2-english"
        )

    def analyze_sentiment(self, texts):
        """
        Analyze sentiment of a list of texts
        """
        try:
            results = self.model(texts)
            return results
        except Exception as e:
            print(f"Error in sentiment analysis: {e}")
            # Return neutral as fallback
            return [{'label': 'NEUTRAL', 'score': 0.5}] * len(texts)


# -------------------------------------------------------------------------
# FUSION MODEL (Random Forest)
# -------------------------------------------------------------------------
class FusionModel:
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)

    def prepare_features(self, technical_pred, sentiment_scores, fundamental_data):
        """
        Prepare input features for fusion model
        """
        pos_sentiment = sum(1 for s in sentiment_scores if s.get('label') == 'POSITIVE') / len(sentiment_scores)
        neg_sentiment = sum(1 for s in sentiment_scores if s.get('label') == 'NEGATIVE') / len(sentiment_scores)

        features = [
            technical_pred,
            pos_sentiment,
            neg_sentiment,
            fundamental_data.get('pe_ratio', 0) or 0,
            fundamental_data.get('pb_ratio', 0) or 0,
            fundamental_data.get('market_cap', 0) or 0
        ]

        return np.array(features).reshape(1, -1)

    def train(self, X, y):
        """
        Train Random Forest fusion model
        """
        self.model.fit(X, y)

    def predict(self, X):
        """
        Predict probability using fusion model
        """
        return self.model.predict_proba(X)


# -------------------------------------------------------------------------
# TEST
# -------------------------------------------------------------------------
if __name__ == "__main__":
    print("TechnicalModel, SentimentModel, FusionModel defined successfully")
