import matplotlib.pyplot as plt
import seaborn as sns

def plot_sentiment_distribution(df):
    """Plot the distribution of sentiment scores."""
    sns.histplot(df["sentiment"], bins=20, kde=True)
    plt.title("Sentiment Score Distribution")
    plt.xlabel("Sentiment Score")
    plt.ylabel("Frequency")
    plt.show()
