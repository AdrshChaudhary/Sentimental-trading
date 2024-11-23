# from src.scraper import scrape_twitter, scrape_reddit, scrape_telegram
from src.scraper import scrape_reddit, scrape_telegram
from src.preprocess import preprocess_data
from src.sentiment import analyze_sentiment
from src.model import train_model
from src.visualizer import visualize_performance
import pandas as pd

def main():
    # Data Scraping
    reddit_data = scrape_reddit("stocks", count=200)
    telegram_data = scrape_telegram("stocks", count=200)

    # Data Preprocessing
    raw_data = pd.concat([reddit_data, telegram_data], ignore_index=True)
    processed_data = preprocess_data(raw_data)

    # Sentiment Analysis
    sentiment_data = analyze_sentiment(processed_data)

    # Model Training
    model, vectorizer, X_test, y_test, metrics = train_model(sentiment_data)
    print(metrics)
    # Model Evaluation
    predictions = model.predict(vectorizer.transform(X_test))

    # Visualization
    visualize_performance(y_test, predictions)

if __name__ == "__main__":
    main()
