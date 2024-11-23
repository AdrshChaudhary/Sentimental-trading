from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

def train_model(data):
    # Ensure that the content column contains only strings
    X = data["content"].astype(str)  # Convert all to strings, just in case
    y = data["sentiment"]

    # Initialize the TF-IDF Vectorizer
    vectorizer = TfidfVectorizer(stop_words='english')

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Fit the vectorizer on the training data and transform it
    X_train_tfidf = vectorizer.fit_transform(X_train)  # Learn vocabulary and transform training data

    # Train a classifier (for example, Naive Bayes)
    model = MultinomialNB()
    model.fit(X_train_tfidf, y_train)

    # Transform the test data
    X_test_tfidf = vectorizer.transform(X_test)

    # Make predictions on the test data
    y_pred = model.predict(X_test_tfidf)

    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model accuracy: {accuracy * 100:.2f}%")

    # Get metrics
    metrics = classification_report(y_test, y_pred)

    return model, vectorizer, X_test, y_test, metrics
