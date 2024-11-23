import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from transformers import pipeline
import pandas as pd

# Download required NLTK resources
nltk.download("stopwords", quiet=True)
nltk.download("punkt", quiet=True)

# Initialize stopwords
STOPWORDS = set(nltk.corpus.stopwords.words("english"))

def clean_text(text, max_length=512):
    """Clean and preprocess text data with length limitation."""
    if not isinstance(text, str):
        return ""  # Return an empty string if the text is not a valid string

    # Remove URLs, mentions, hashtags, and punctuation
    text = re.sub(r"http\S+|www\S+|https\S+", "", text, flags=re.MULTILINE)
    text = re.sub(r"\@\w+|\#", "", text)
    text = re.sub(r"[^\w\s]", "", text)
    text = text.lower()  # Convert to lowercase

    # Tokenize and remove stopwords
    words = [word for word in word_tokenize(text) if word not in STOPWORDS]
    
    # Truncate to max length
    truncated_text = " ".join(words[:max_length])
    return truncated_text

def preprocess_data(df):
    """Apply text cleaning and sentiment analysis to a DataFrame."""
    # Ensure DataFrame has the correct structure
    if not isinstance(df, pd.DataFrame):
        raise ValueError("Input must be a pandas DataFrame")
    
    # Ensure 'content' column exists
    if 'content' not in df.columns:
        raise ValueError("DataFrame must have a 'content' column")

    # Clean the text data
    df['cleaned_content'] = df['content'].apply(clean_text)

    # Initialize sentiment analyzer
    sentiment_analyzer = pipeline("sentiment-analysis", 
                                  model="distilbert-base-uncased-finetuned-sst-2-english", 
                                  truncation=True, 
                                  max_length=512)

    # Batch sentiment analysis to avoid memory issues
    def batch_sentiment_analysis(texts, batch_size=32):
        sentiments = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            try:
                batch_sentiments = sentiment_analyzer(batch)
                sentiments.extend([s['label'] for s in batch_sentiments])
            except Exception as e:
                print(f"Error processing batch: {e}")
                # Fallback to individual processing if batch fails
                for text in batch:
                    try:
                        sentiment = sentiment_analyzer(text)[0]['label']
                        sentiments.append(sentiment)
                    except Exception as e:
                        print(f"Error processing individual text: {e}")
                        sentiments.append('NEUTRAL')
        return sentiments

    # Apply sentiment analysis in batches
    df['sentiment'] = batch_sentiment_analysis(df['cleaned_content'].tolist())

    # Map sentiment to binary values
    sentiment_mapping = {"NEGATIVE": 0, "POSITIVE": 1, "NEUTRAL": 0}
    df['sentiment'] = df['sentiment'].map(sentiment_mapping)

    # Drop rows where sentiment couldn't be determined
    df = df.dropna(subset=['sentiment'])

    return df