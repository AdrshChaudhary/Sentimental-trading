import logging
import praw
from telethon.sync import TelegramClient
import pandas as pd
import streamlit as st

logging.basicConfig(level=logging.INFO)

def get_api_credentials():
    """Get API credentials from Streamlit secrets"""
    try:
        credentials = {
            'REDDIT_CLIENT_ID': st.secrets['api_credentials']['REDDIT_CLIENT_ID'],
            'REDDIT_SECRET': st.secrets['api_credentials']['REDDIT_SECRET'],
            'REDDIT_USER_AGENT': st.secrets['api_credentials']['REDDIT_USER_AGENT'],
            'REDDIT_USERNAME': st.secrets['api_credentials']['REDDIT_USERNAME'],
            'REDDIT_PASSWORD': st.secrets['api_credentials']['REDDIT_PASSWORD'],
            'TELEGRAM_API_ID': st.secrets['api_credentials']['TELEGRAM_API_ID'],
            'TELEGRAM_API_HASH': st.secrets['api_credentials']['TELEGRAM_API_HASH']
        }
        return credentials
    except Exception as e:
        logging.error(f"Error fetching credentials: {e}")
        return {}

def get_credential(credentials, key):
    """Safely get credential value with error handling"""
    if not credentials:
        raise ValueError(f"No credentials available")
    
    value = credentials.get(key)
    if not value:
        raise ValueError(f"Missing credential: {key}")
    
    return value

def scrape_reddit(subreddit_name, count=100):
    """Scrape posts from a subreddit using Streamlit secrets."""
    try:
        # Get credentials from Streamlit secrets
        credentials = get_api_credentials()
        
        if not credentials:
            logging.warning("No Reddit credentials found. Returning empty DataFrame.")
            return pd.DataFrame(columns=["title", "content"])
        
        # Initialize Reddit client with credentials from secrets
        reddit = praw.Reddit(
            client_id=get_credential(credentials, 'REDDIT_CLIENT_ID'),
            client_secret=get_credential(credentials, 'REDDIT_SECRET'),
            user_agent=get_credential(credentials, 'REDDIT_USER_AGENT'),
            username=get_credential(credentials, 'REDDIT_USERNAME'),
            password=get_credential(credentials, 'REDDIT_PASSWORD')
        )
        
        # Scrape subreddit
        subreddit = reddit.subreddit(subreddit_name)
        posts = []
        
        try:
            for post in subreddit.hot(limit=count):
                posts.append({
                    "title": post.title,
                    "content": post.selftext
                })
        except Exception as e:
            logging.error(f"Error while fetching posts: {e}")
        
        if not posts:
            logging.warning(f"No posts found in subreddit: {subreddit_name}")
            return pd.DataFrame(columns=["title", "content"])
            
        return pd.DataFrame(posts)
        
    except Exception as e:
        logging.error(f"Reddit scraping error: {e}")
        return pd.DataFrame(columns=["title", "content"])

def scrape_telegram(channel_name, count=100):
    """Scrape messages from a Telegram channel using Streamlit secrets."""
    try:
        # Get credentials from Streamlit secrets
        credentials = get_api_credentials()
        
        if not credentials:
            logging.warning("No Telegram credentials found. Returning empty DataFrame.")
            return pd.DataFrame(columns=["platform", "content"])
        
        # Get API credentials
        api_id = get_credential(credentials, 'TELEGRAM_API_ID')
        api_hash = get_credential(credentials, 'TELEGRAM_API_HASH')
        
        messages = []
        with TelegramClient("anon", int(api_id), api_hash) as client:
            try:
                for message in client.iter_messages(channel_name, limit=count):
                    if message.text:
                        messages.append({
                            "platform": "telegram",
                            "content": message.text
                        })
            except Exception as e:
                logging.error(f"Error while fetching messages: {e}")
        
        if not messages:
            logging.warning(f"No messages found in channel: {channel_name}")
            return pd.DataFrame(columns=["platform", "content"])
            
        return pd.DataFrame(messages)
        
    except Exception as e:
        logging.error(f"Telegram scraping error: {e}")
        return pd.DataFrame(columns=["platform", "content"])