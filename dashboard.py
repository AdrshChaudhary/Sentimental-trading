import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from textblob import TextBlob
import plotly.express as px
import plotly.graph_objs as go
from datetime import datetime, timedelta
import ta
from src.scraper import scrape_reddit, scrape_telegram
from src.preprocess import preprocess_data
from src.model import train_model
import requests
import yfinance as yf

def fetch_stock_data(ticker, period='1y'):
    """Fetch stock data using yfinance"""
    try:
        stock = yf.Ticker(ticker)
        df = stock.history(period=period)
        return stock, df
    except Exception as e:
        st.error(f"Error fetching stock data: {str(e)}")
        return None, None

def calculate_technical_indicators(df):
    """Calculate technical indicators"""
    if len(df) > 0:
        # Moving averages
        df['SMA_20'] = ta.trend.sma_indicator(df['Close'], window=20)
        df['SMA_50'] = ta.trend.sma_indicator(df['Close'], window=50)
        df['EMA_20'] = ta.trend.ema_indicator(df['Close'], window=20)
        
        # RSI
        df['RSI'] = ta.momentum.rsi(df['Close'], window=14)
        
        # MACD
        macd = ta.trend.MACD(df['Close'])
        df['MACD'] = macd.macd()
        df['MACD_Signal'] = macd.macd_signal()
        df['MACD_Hist'] = macd.macd_diff()
        
        # Bollinger Bands
        bollinger = ta.volatility.BollingerBands(df['Close'])
        df['BB_High'] = bollinger.bollinger_hband()
        df['BB_Low'] = bollinger.bollinger_lband()
        df['BB_Mid'] = bollinger.bollinger_mavg()
    
    return df

def get_stock_info_safe(stock):
    """Safely get stock information with fallbacks"""
    info = stock.info
    return {
        'currentPrice': info.get('currentPrice', info.get('regularMarketPrice', 'N/A')),
        'marketCap': info.get('marketCap', 'N/A'),
        'fiftyTwoWeekChange': info.get('52WeekChange', info.get('fiftyTwoWeekChange', 'N/A')),
        'trailingPE': info.get('trailingPE', 'N/A'),
        'forwardPE': info.get('forwardPE', 'N/A'),
        'pegRatio': info.get('pegRatio', 'N/A'),
        'priceToBook': info.get('priceToBook', 'N/A'),
        'revenueGrowth': info.get('revenueGrowth', 'N/A'),
        'earningsGrowth': info.get('earningsGrowth', 'N/A'),
        'profitMargins': info.get('profitMargins', 'N/A'),
        'operatingMargins': info.get('operatingMargins', 'N/A')
    }

def fetch_crypto_data(symbol, period='1y'):
    """Fetch cryptocurrency data using yfinance"""
    try:
        # Add '-USD' suffix for cryptocurrency pairs
        crypto_ticker = f"{symbol}-USD"
        crypto = yf.Ticker(crypto_ticker)
        df = crypto.history(period=period)
        return crypto, df
    except Exception as e:
        st.error(f"Error fetching crypto data: {str(e)}")
        return None, None
    
def get_news(ticker):
    """Fetch news for the given ticker"""
    stock = yf.Ticker(ticker)
    return stock.news

def create_enhanced_dashboard(processed_data):
    st.set_page_config(page_title="Advanced Stock & Crypto Analysis Dashboard", layout="wide")
    
    # Custom CSS
    st.markdown("""
    <style>
    .main-header {
        font-size: 40px;
        font-weight: bold;
        color: #2C3E50;
        text-align: center;
        margin-bottom: 30px;
    }
    .metric-card {
        background-color: #f7f7f7;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.1);
    }
    .news-card {
        background-color: white;
        padding: 15px;
        border-radius: 5px;
        margin: 10px 0;
        border: 1px solid #eee;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("Settings")
    analysis_type = st.sidebar.selectbox("Select Analysis Type", ["Stocks", "Crypto", "Sentiment"])
    
    if analysis_type == "Stocks":
        ticker = st.sidebar.text_input("Enter Stock Ticker", "AAPL")
        period = st.sidebar.selectbox("Select Time Period", ['1mo', '3mo', '6mo', '1y', '2y', '5y'])
        
        # Main Content
        st.markdown("<h1 class='main-header'>📊 Stock Analysis Dashboard</h1>", unsafe_allow_html=True)
        
        # Create tabs
        tab1, tab2, tab3, tab4 = st.tabs(["Overview", "Technical Analysis", "Financial Metrics", "News"])
        
        try:
            stock, df = fetch_stock_data(ticker, period)
            if stock and df is not None and not df.empty:
                df = calculate_technical_indicators(df)
                info = get_stock_info_safe(stock)
            
                with tab1:
                    # Stock Info
                    info = stock.info
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        current_price = info['currentPrice']
                        if current_price != 'N/A':
                            st.metric("Current Price", f"${current_price:,.2f}")
                        else:
                            st.metric("Current Price", "N/A")
                    
                    with col2:
                        market_cap = info['marketCap']
                        if market_cap != 'N/A':
                            st.metric("Market Cap", f"${market_cap:,.0f}")
                        else:
                            st.metric("Market Cap", "N/A")
                    
                    with col3:
                        week_change = info['fiftyTwoWeekChange']
                        if week_change != 'N/A':
                            st.metric("52 Week Change", f"{week_change:.2%}")
                        else:
                            st.metric("52 Week Change", "N/A")
                    # Price Chart
                    fig = go.Figure()
                    fig.add_trace(go.Candlestick(
                        x=df.index,
                        open=df['Open'],
                        high=df['High'],
                        low=df['Low'],
                        close=df['Close'],
                        name='OHLC'
                    ))
                    fig.update_layout(title=f'{ticker} Stock Price', xaxis_title='Date', yaxis_title='Price')
                    st.plotly_chart(fig, use_container_width=True)
            
                with tab2:
                    # Technical Indicators
                    st.subheader("Technical Indicators")
                    
                    # Moving Averages
                    fig_ma = go.Figure()
                    fig_ma.add_trace(go.Scatter(x=df.index, y=df['Close'], name='Price'))
                    fig_ma.add_trace(go.Scatter(x=df.index, y=df['SMA_20'], name='SMA 20'))
                    fig_ma.add_trace(go.Scatter(x=df.index, y=df['SMA_50'], name='SMA 50'))
                    fig_ma.update_layout(title='Moving Averages')
                    st.plotly_chart(fig_ma, use_container_width=True)
                    
                    # RSI
                    fig_rsi = go.Figure()
                    fig_rsi.add_trace(go.Scatter(x=df.index, y=df['RSI'], name='RSI'))
                    fig_rsi.add_hline(y=70, line_dash="dash", line_color="red")
                    fig_rsi.add_hline(y=30, line_dash="dash", line_color="green")
                    fig_rsi.update_layout(title='Relative Strength Index (RSI)')
                    st.plotly_chart(fig_rsi, use_container_width=True)
                    
                    # MACD
                    fig_macd = go.Figure()
                    fig_macd.add_trace(go.Scatter(x=df.index, y=df['MACD'], name='MACD'))
                    fig_macd.add_trace(go.Scatter(x=df.index, y=df['MACD_Signal'], name='Signal Line'))
                    fig_macd.update_layout(title='MACD')
                    fig_macd.add_trace(go.Bar(x=df.index, y=df['MACD_Hist'], name='MACD Histogram'))
                    st.plotly_chart(fig_macd, use_container_width=True)
                
                with tab3:
                    # Financial Metrics
                    st.subheader("Financial Metrics")
                    
                    metrics_col1, metrics_col2 = st.columns(2)
                    
                    with metrics_col1:
                        st.markdown("### Valuation Metrics")
                        metrics = {
                            "P/E Ratio": info.get('trailingPE', 'N/A'),
                            "Forward P/E": info.get('forwardPE', 'N/A'),
                            "PEG Ratio": info.get('pegRatio', 'N/A'),
                            "Price to Book": info.get('priceToBook', 'N/A')
                        }
                        
                        for metric, value in metrics.items():
                            if isinstance(value, float):
                                st.metric(metric, f"{value:.2f}")
                            else:
                                st.metric(metric, value)
                    
                    with metrics_col2:
                        st.markdown("### Growth Metrics")
                        growth_metrics = {
                            "Revenue Growth": info.get('revenueGrowth', 'N/A'),
                            "Earnings Growth": info.get('earningsGrowth', 'N/A'),
                            "Profit Margin": info.get('profitMargins', 'N/A'),
                            "Operating Margin": info.get('operatingMargins', 'N/A')
                        }
                        
                        for metric, value in growth_metrics.items():
                            if isinstance(value, float):
                                st.metric(metric, f"{value:.2%}")
                            else:
                                st.metric(metric, value)
                
                # with tab4:
                #     # News
                #     st.subheader("Latest News")
                #     news = get_news(ticker)
                    
                #     for article in news[:5]:
                #         with st.container():
                #             st.markdown(f"""
                #             <div class="news-card">
                #                 <h4>{article['title']}</h4>
                #                 <p>{article['description']}</p>
                #                 <small>Source: {article['source']} | {datetime.fromtimestamp(article['providerPublishTime']).strftime('%Y-%m-%d %H:%M:%S')}</small>
                #             </div>
                #             """, unsafe_allow_html=True)
        
        except Exception as e:
            st.error(f"Error fetching data for {ticker}: {str(e)}")
    
    elif analysis_type == "Crypto":
        crypto_symbol = st.sidebar.text_input("Enter Crypto Symbol", "BTC")
        period = st.sidebar.selectbox("Select Time Period", ['1mo', '3mo', '6mo', '1y', '2y', '5y'])
        
        st.markdown("<h1 class='main-header'>🪙 Cryptocurrency Analysis</h1>", unsafe_allow_html=True)
        
        # Create tabs for crypto analysis
        crypto_tab1, crypto_tab2, crypto_tab3 = st.tabs(["Overview", "Technical Analysis", "Market Data"])
        
        try:
            crypto, df = fetch_crypto_data(crypto_symbol, period)
            
            if df is not None and not df.empty:
                df = calculate_technical_indicators(df)  # Reuse the same technical indicators
                
                with crypto_tab1:
                    # Price and Volume Overview
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        current_price = df['Close'].iloc[-1]
                        prev_price = df['Close'].iloc[-2]
                        price_change = ((current_price - prev_price) / prev_price) * 100
                        st.metric("Current Price", f"${current_price:,.2f}", 
                                f"{price_change:+.2f}%")
                    
                    with col2:
                        volume = df['Volume'].iloc[-1]
                        st.metric("24h Volume", f"${volume:,.0f}")
                    
                    with col3:
                        price_change_7d = ((current_price - df['Close'].iloc[-7]) / df['Close'].iloc[-7]) * 100
                        st.metric("7D Change", f"{price_change_7d:+.2f}%")
                    
                    # Candlestick Chart
                    fig = go.Figure()
                    fig.add_trace(go.Candlestick(
                        x=df.index,
                        open=df['Open'],
                        high=df['High'],
                        low=df['Low'],
                        close=df['Close'],
                        name='OHLC'
                    ))
                    fig.update_layout(title=f'{crypto_symbol} Price', xaxis_title='Date', yaxis_title='Price (USD)')
                    st.plotly_chart(fig, use_container_width=True)
                
                with crypto_tab2:
                    st.subheader("Technical Indicators")
                    
                    # Moving Averages
                    fig_ma = go.Figure()
                    fig_ma.add_trace(go.Scatter(x=df.index, y=df['Close'], name='Price'))
                    fig_ma.add_trace(go.Scatter(x=df.index, y=df['SMA_20'], name='SMA 20'))
                    fig_ma.add_trace(go.Scatter(x=df.index, y=df['SMA_50'], name='SMA 50'))
                    fig_ma.update_layout(title='Moving Averages')
                    st.plotly_chart(fig_ma, use_container_width=True)
                    
                    # RSI
                    fig_rsi = go.Figure()
                    fig_rsi.add_trace(go.Scatter(x=df.index, y=df['RSI'], name='RSI'))
                    fig_rsi.add_hline(y=70, line_dash="dash", line_color="red")
                    fig_rsi.add_hline(y=30, line_dash="dash", line_color="green")
                    fig_rsi.update_layout(title='Relative Strength Index (RSI)')
                    st.plotly_chart(fig_rsi, use_container_width=True)
                    
                    # MACD
                    fig_macd = go.Figure()
                    fig_macd.add_trace(go.Scatter(x=df.index, y=df['MACD'], name='MACD'))
                    fig_macd.add_trace(go.Scatter(x=df.index, y=df['MACD_Signal'], name='Signal Line'))
                    fig_macd.update_layout(title='MACD')
                    st.plotly_chart(fig_macd, use_container_width=True)
                
                with crypto_tab3:
                    st.subheader("Market Statistics")
                    
                    # Calculate additional statistics
                    stats_col1, stats_col2 = st.columns(2)
                    
                    with stats_col1:
                        st.metric("All-Time High", f"${df['High'].max():,.2f}")
                        st.metric("All-Time Low", f"${df['Low'].min():,.2f}")
                        st.metric("Average Daily Volume", f"${df['Volume'].mean():,.0f}")
                    
                    with stats_col2:
                        # Calculate volatility (standard deviation of returns)
                        returns = df['Close'].pct_change()
                        volatility = returns.std() * np.sqrt(252)  # Annualized volatility
                        st.metric("Volatility (Annualized)", f"{volatility:.2%}")
                        
                        # Calculate Sharpe Ratio (assuming risk-free rate of 0.01)
                        risk_free_rate = 0.01
                        excess_returns = returns - risk_free_rate/252
                        sharpe_ratio = np.sqrt(252) * excess_returns.mean() / returns.std()
                        st.metric("Sharpe Ratio", f"{sharpe_ratio:.2f}")
                        
                        # Maximum Drawdown
                        rolling_max = df['Close'].cummax()
                        drawdown = (df['Close'] - rolling_max) / rolling_max
                        max_drawdown = drawdown.min()
                        st.metric("Maximum Drawdown", f"{max_drawdown:.2%}")
                    
                    # Volume Analysis
                    fig_volume = go.Figure()
                    fig_volume.add_trace(go.Bar(x=df.index, y=df['Volume'], name='Volume'))
                    fig_volume.update_layout(title='Trading Volume History')
                    st.plotly_chart(fig_volume, use_container_width=True)
            
            else:
                st.error(f"No data available for {crypto_symbol}. Please check the symbol and try again.")
        
        except Exception as e:
            st.error(f"Error analyzing cryptocurrency: {str(e)}")

    
    else:  
        st.markdown("<h1 class='main-header'>🎭 Sentiment Analysis Dashboard</h1>", unsafe_allow_html=True)
        tab1, tab2, tab3, tab4 = st.tabs([
            "Sentiment Overview", 
            "Detailed Analysis", 
            "Source Comparison", 
            "Advanced Insights"
        ])
        
        with tab1:
            st.subheader("Overall Sentiment Distribution")
            if 'sentiment' in processed_data.columns:
                sentiment_counts = processed_data['sentiment'].value_counts()
                col1, col2 = st.columns(2)
                
                with col1:
                    fig_pie = px.pie(
                        values=sentiment_counts.values, 
                        names=sentiment_counts.index.map({0: 'Negative', 1: 'Positive'}),
                        title='Sentiment Breakdown',
                        color_discrete_sequence=['#FF6B6B', '#4ECDC4']
                    )
                    st.plotly_chart(fig_pie)
                
                with col2:
                    st.metric("Total Posts Analyzed", len(processed_data))
                    st.metric("Positive Sentiment", f"{sentiment_counts.get(1, 0)} posts")
                    st.metric("Negative Sentiment", f"{sentiment_counts.get(0, 0)} posts")
        
        with tab2:
            # Detailed Text Analysis
            st.subheader("Detailed Sentiment Exploration")
            
            # Add text polarity and subjectivity
            if 'content' in processed_data.columns:
                processed_data['polarity'] = processed_data['content'].apply(
                    lambda x: TextBlob(str(x)).sentiment.polarity if pd.notnull(x) else 0
                )
                processed_data['subjectivity'] = processed_data['content'].apply(
                    lambda x: TextBlob(str(x)).sentiment.subjectivity if pd.notnull(x) else 0
                )
                
                # Scatter plot
                fig_scatter = px.scatter(
                    processed_data,
                    x='polarity',
                    y='subjectivity',
                    color='sentiment',
                    title='Sentiment Polarity vs Subjectivity',
                    labels={'sentiment': 'Sentiment'},
                    color_discrete_map={0: '#FF6B6B', 1: '#4ECDC4'}
                )
                st.plotly_chart(fig_scatter)
            
            # Top positive and negative texts
            st.subheader("Sample Texts")
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("Top Positive Texts")
                top_positive = processed_data[processed_data['sentiment'] == 1].nlargest(5, 'polarity')
                for _, row in top_positive.iterrows():
                    st.text(row['content'][:200] + '...')
            
            with col2:
                st.write("Top Negative Texts")
                top_negative = processed_data[processed_data['sentiment'] == 0].nsmallest(5, 'polarity')
                for _, row in top_negative.iterrows():
                    st.text(row['content'][:200] + '...')
        
        with tab3:
            # Source Comparison if multiple sources exist
            if 'platform' in processed_data.columns:
                st.subheader("Sentiment by Source Platform")
                platform_sentiment = processed_data.groupby('platform')['sentiment'].value_counts(normalize=True).unstack()
                
                fig_bar = px.bar(
                    platform_sentiment, 
                    title='Sentiment Distribution by Platform',
                    labels={'value': 'Percentage', 'platform': 'Platform'}
                )
                st.plotly_chart(fig_bar)
        
        with tab4:
            # Advanced Insights
            st.subheader("Advanced Sentiment Insights")
            
            # Word Cloud (if you have wordcloud library)
            # This would require additional processing
            
            # Time-based sentiment if timestamp available
            if 'timestamp' in processed_data.columns:
                processed_data['date'] = pd.to_datetime(processed_data['timestamp']).dt.date
                daily_sentiment = processed_data.groupby('date')['sentiment'].mean()
                
                fig_line = go.Figure()
                fig_line.add_trace(go.Scatter(
                    x=daily_sentiment.index, 
                    y=daily_sentiment.values, 
                    mode='lines+markers'
                ))
                fig_line.update_layout(title='Sentiment Trend Over Time')
                st.plotly_chart(fig_line)

def main():
    # Load data
    reddit_data = scrape_reddit("stocks", count=200)
    telegram_data = scrape_telegram("stocks", count=200)
    
    # Preprocess data
    raw_data = pd.concat([reddit_data, telegram_data], ignore_index=True)
    processed_data = preprocess_data(raw_data)
    
    # Create dashboard
    create_enhanced_dashboard(processed_data)

if __name__ == "__main__":
    main()
