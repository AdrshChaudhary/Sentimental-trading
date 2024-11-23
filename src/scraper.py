import os
import logging
import praw
from telethon.sync import TelegramClient
from dotenv import load_dotenv
import pandas as pd

load_dotenv()
logging.basicConfig(level=logging.INFO)

def get_env_var(var_name, default=None):
    value = os.getenv(var_name, default)
    if value is None:
        logging.error(f"Missing environment variable: {var_name}")
        raise ValueError(f"Environment variable {var_name} is required")
    return value

def scrape_reddit(subreddit_name, count=100):
    """Scrape posts from a subreddit with error handling."""
    try:
        reddit = praw.Reddit(
            client_id=get_env_var("REDDIT_CLIENT_ID"),
            client_secret=get_env_var("REDDIT_SECRET"),
            user_agent=get_env_var("REDDIT_USER_AGENT"),
            username=get_env_var("REDDIT_USERNAME"),
            password=get_env_var("REDDIT_PASSWORD")
        )
        subreddit = reddit.subreddit(subreddit_name)
        posts = [(post.title, post.selftext) for post in subreddit.hot(limit=count)]
        return pd.DataFrame(posts, columns=["title", "content"])
    except Exception as e:
        logging.error(f"Reddit scraping error: {e}")
        return pd.DataFrame(columns=["title", "content"])

def scrape_telegram(channel_name, count=100):
    """Scrape messages from a Telegram channel with error handling."""
    try:
        api_id = get_env_var("TELEGRAM_API_ID")
        api_hash = get_env_var("TELEGRAM_API_HASH")
        
        with TelegramClient("anon", api_id, api_hash) as client:
            messages = [message.text for message in client.iter_messages(channel_name, limit=count) if message.text]
        
        return pd.DataFrame({"platform": "telegram", "content": messages})
    except Exception as e:
        logging.error(f"Telegram scraping error: {e}")
        return pd.DataFrame(columns=["platform", "content"])