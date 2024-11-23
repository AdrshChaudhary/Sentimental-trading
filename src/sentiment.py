from transformers import pipeline
import logging

def analyze_sentiment(data, model_name="distilbert-base-uncased-finetuned-sst-2-english"):
    """Centralized sentiment analysis with improved error handling."""
    try:
        sentiment_analyzer = pipeline("sentiment-analysis", model=model_name)
        
        def get_sentiment(text):
            try:
                result = sentiment_analyzer(text[:512])  # Truncate to avoid memory issues
                return result[0]["label"]
            except Exception as e:
                logging.warning(f"Sentiment analysis error for text: {e}")
                return "NEUTRAL"
        
        data["sentiment"] = data["content"].apply(get_sentiment)
        sentiment_mapping = {"NEGATIVE": 0, "POSITIVE": 1, "NEUTRAL": 0}
        data["sentiment"] = data["sentiment"].map(sentiment_mapping)
        
        return data
    except Exception as e:
        logging.error(f"Sentiment analysis failed: {e}")
        return data