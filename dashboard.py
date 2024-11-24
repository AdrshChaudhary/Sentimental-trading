import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objs as go
from datetime import datetime
import asyncio
from src.scraper import scrape_reddit, scrape_telegram
from src.preprocess import preprocess_data
from src.model import train_model
from src.sentiment import analyze_sentiment
import ta
import yfinance as yf
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Set page config at the very beginning
st.set_page_config(
    page_title="Advanced Financial Analysis Dashboard",
    layout="wide",
    initial_sidebar_state="collapsed"
)


@st.cache_data(ttl=3600)
def fetch_data(symbol, period='1y', is_crypto=False):
    """Fetch stock or crypto data using yfinance"""
    try:
        if is_crypto:
            symbol = f"{symbol}-USD"
        
        # Add error handling for invalid symbols
        if not symbol or len(symbol.strip()) == 0:
            st.error("Please enter a valid symbol")
            return None
            
        data = yf.download(symbol, period=period)
        
        if data.empty:
            st.error(f"No data found for symbol: {symbol}")
            return None
            
        # Convert multi-level columns to single level if necessary
        if isinstance(data.columns, pd.MultiIndex):
            # Flatten column names using the first level
            data.columns = [col[0] for col in data.columns]
            
        # Validate that we have the required columns
        required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        missing_columns = [col for col in required_columns if col not in data.columns]
        
        if missing_columns:
            st.error(f"Missing required data columns: {', '.join(missing_columns)}")
            return None
            
        return data
        
    except Exception as e:
        st.error(f"Error fetching data: {str(e)}")
        return None
        
    except Exception as e:
        st.error(f"Error fetching data: {str(e)}")
        return None

@st.cache_data(ttl=3600)

def calculate_technical_indicators(df):
    """Calculate technical indicators"""
    if len(df) > 0:

        if isinstance(df['Close'], pd.DataFrame):  # If it's a DataFrame, flatten it
            close_prices = df['Close'].iloc[:, 0]  # Use the first column
        else:
            close_prices = df['Close']

        # Convert to 1D Series if it's still not 1D
        close_prices = pd.Series(close_prices.values.flatten(), index=df.index, name='Close')
        # Moving averages
        df['SMA_20'] = ta.trend.sma_indicator(close=close_prices, window=20)
        df['SMA_50'] = ta.trend.sma_indicator(close=close_prices, window=50)
        df['EMA_20'] = ta.trend.ema_indicator(close=close_prices, window=20)

        # RSI
        df['RSI'] = ta.momentum.rsi(close=close_prices, window=14)

        # Bollinger Bands
        bollinger = ta.volatility.BollingerBands(close=close_prices)
        df['BB_High'] = bollinger.bollinger_hband()
        df['BB_Low'] = bollinger.bollinger_lband()
        df['BB_Mid'] = bollinger.bollinger_mavg()
    
    return df

@st.cache_data(ttl=3600)
def fetch_financial_news(symbol):
    """Fetch news from multiple sources"""
    news_list = []
    
    # Fetch from Yahoo Finance
    try:
        ticker = yf.Ticker(symbol)
        yahoo_news = ticker.news
        if yahoo_news:
            for article in yahoo_news[:5]:
                news_list.append({
                    'title': article.get('title'),
                    'description': article.get('summary'),
                    'source': 'Yahoo Finance',
                    'link': article.get('link'),
                    'published': datetime.fromtimestamp(article.get('providerPublishTime', 0))
                })
    except Exception as e:
        print(f"Error fetching Yahoo Finance news: {e}")

    return news_list

@st.cache_data
def get_asset_info(symbol, is_crypto=False):
    """Safely get asset information with fallbacks"""
    try:
        ticker = yf.Ticker(symbol)
        info = ticker.info
        
        if is_crypto:
            return {
                'currentPrice': info.get('regularMarketPrice', 'N/A'),
                'marketCap': info.get('marketCap', 'N/A'),
                'volume24h': info.get('volume24h', info.get('regularMarketVolume', 'N/A')),
                'circulatingSupply': info.get('circulatingSupply', 'N/A'),
                'totalSupply': info.get('totalSupply', 'N/A'),
                'maxSupply': info.get('maxSupply', 'N/A')
            }
        else:
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
    except Exception as e:
        st.error(f"Error fetching asset info: {str(e)}")
        return {}

@st.cache_data(ttl=3600)
def get_processed_data():
    """Process and analyze sentiment data"""
    try:

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        # Data Scraping
        reddit_data = scrape_reddit("stocks", count=200)
        telegram_data = scrape_telegram("stocks", count=200)

        loop.close()

        # Data Preprocessing
        raw_data = pd.concat([reddit_data, telegram_data], ignore_index=True)
        processed_data = preprocess_data(raw_data)

        # Sentiment Analysis
        sentiment_data = analyze_sentiment(processed_data)

        # Model Training
        model, vectorizer, X_test, y_test, metrics = train_model(sentiment_data)
        
        # Make predictions
        predictions = model.predict(vectorizer.transform(X_test))

        return sentiment_data, metrics, y_test, predictions
    except Exception as e:
        st.error(f"Error processing data: {str(e)}")
        return None, None, None, None

def create_price_chart(df, symbol):
    """Create candlestick chart with Bollinger Bands."""
    try:
        # Define required columns
        required_columns = ['Open', 'High', 'Low', 'Close']
        
        # Check if df is None or empty
        if df is None or df.empty:
            st.error("No data available for the selected symbol")
            return go.Figure()
        
        # Convert multi-level columns to single level if necessary
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [col[0] for col in df.columns]
        
        # Check if all required columns exist
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            st.error(f"Missing required columns: {', '.join(missing_columns)}")
            return go.Figure()
        
        # Create a copy of the DataFrame to avoid modifying the original
        chart_df = df.copy()
        
        # Handle any NaN values
        chart_df = chart_df.dropna(subset=required_columns)
        
        if len(chart_df) == 0:
            st.error("No valid data points after removing missing values")
            return go.Figure()
        
        # Convert columns to numeric
        for col in required_columns:
            chart_df[col] = pd.to_numeric(chart_df[col], errors='coerce')
        
        # Create figure
        fig = go.Figure()
        
        # Add candlestick chart
        fig.add_trace(go.Candlestick(
            x=chart_df.index,
            open=chart_df['Open'],
            high=chart_df['High'],
            low=chart_df['Low'],
            close=chart_df['Close'],
            name='OHLC',
            increasing_line_color='#26a69a',
            decreasing_line_color='#ef5350'
        ))
        
        # Add Bollinger Bands if available
        bb_columns = ['BB_High', 'BB_Low', 'BB_Mid']
        if all(col in chart_df.columns for col in bb_columns):
            for band in ['BB_High', 'BB_Low']:
                fig.add_trace(go.Scatter(
                    x=chart_df.index,
                    y=chart_df[band],
                    name=f'BB {"Upper" if band == "BB_High" else "Lower"}',
                    line=dict(color='gray', dash='dash'),
                    opacity=0.7
                ))
            
            fig.add_trace(go.Scatter(
                x=chart_df.index,
                y=chart_df['BB_Mid'],
                name='BB Middle',
                line=dict(color='blue', dash='dash'),
                opacity=0.7
            ))
        
        # Update layout
        fig.update_layout(
            title=f'{symbol} Price Chart',
            yaxis_title='Price',
            xaxis_title='Date',
            template='plotly_white',
            height=600,
            showlegend=True,
            xaxis_rangeslider_visible=False
        )
        
        return fig
        
    except Exception as e:
        st.error(f"Error creating price chart: {str(e)}")
        return go.Figure()

def create_technical_charts(df):
    """Create technical analysis charts."""
    # Moving Averages with Volume
    fig_ma = go.Figure()
    fig_ma.add_trace(go.Scatter(x=df.index, y=df['Close'], name='Price'))
    if 'SMA_20' in df and 'SMA_50' in df:
        fig_ma.add_trace(go.Scatter(x=df.index, y=df['SMA_20'], name='SMA 20'))
        fig_ma.add_trace(go.Scatter(x=df.index, y=df['SMA_50'], name='SMA 50'))
    fig_ma.add_trace(go.Bar(x=df.index, y=df['Volume'], name='Volume', yaxis='y2', opacity=0.3))
    fig_ma.update_layout(
        title='Price, Moving Averages & Volume',
        yaxis2=dict(title='Volume', overlaying='y', side='right'),
        height=400
    )

    # RSI
    fig_rsi = go.Figure()
    if 'RSI' in df:
        fig_rsi.add_trace(go.Scatter(x=df.index, y=df['RSI'], name='RSI'))
        fig_rsi.add_hline(y=70, line_dash="dash", line_color="red")
        fig_rsi.add_hline(y=30, line_dash="dash", line_color="green")
        fig_rsi.update_layout(
        title='Relative Strength Index (RSI)',
        yaxis_title='RSI',
        height=400
        )

    return fig_ma, fig_rsi



def sentiment_analysis(processed_data, metrics, y_test, predictions):
    """Create enhanced sentiment analysis visualizations"""
    st.subheader("Sentiment Analysis Results")
    
    if metrics:
        # Parse classification report string to extract metrics
        metrics_dict = {}
        for line in metrics.split('\n'):
            if 'accuracy' in line:
                metrics_dict['accuracy'] = float(line.split()[-1])
            elif 'weighted avg' in line:
                parts = line.split()
                metrics_dict['precision'] = float(parts[-3])
                metrics_dict['recall'] = float(parts[-2])
        
        # Display metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Model Accuracy", f"{metrics_dict.get('accuracy', 0):.2%}")
        with col2:
            st.metric("Precision", f"{metrics_dict.get('precision', 0):.2%}")
        with col3:
            st.metric("Recall", f"{metrics_dict.get('recall', 0):.2%}")
    
    # Sentiment Distribution
    if 'sentiment' in processed_data.columns:
        sentiment_counts = processed_data['sentiment'].value_counts()
        fig_sentiment = px.pie(
            values=sentiment_counts.values,
            names=['Negative', 'Positive'],
            title='Sentiment Distribution',
            color_discrete_sequence=['#FF6B6B', '#4ECDC4']
        )
        st.plotly_chart(fig_sentiment)

        # Create word cloud
        if 'content' in processed_data.columns:
            text = ' '.join(processed_data['content'].astype(str))
            wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
            
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.imshow(wordcloud, interpolation='bilinear')
            ax.axis('off')
            st.pyplot(fig)
            plt.close()

            # Show sample texts
            st.subheader("Sample Analyzed Texts")
            col1, col2 = st.columns(2)
    with col1:
        st.write("Most Positive Texts")
        positive_texts = processed_data[processed_data['sentiment'] == 1].head()
        for idx, row in positive_texts.iterrows():
            st.text_area(
                label=f"Positive Text {idx+1}",
                value=row['content'][:200] + "...",
                height=100,
                label_visibility="collapsed"
            )
    
    with col2:
        st.write("Most Negative Texts")
        negative_texts = processed_data[processed_data['sentiment'] == 0].head()
        for idx, row in negative_texts.iterrows():
            st.text_area(
                label=f"Negative Text {idx+1}",
                value=row['content'][:200] + "...",
                height=100,
                label_visibility="collapsed"
            )


def calculate_statistics(df):
    """Calculate additional market statistics."""
    # Ensure 'Close' and other columns are 1D
    high_prices = df['High'].values.flatten() if isinstance(df['High'], pd.DataFrame) else df['High']
    low_prices = df['Low'].values.flatten() if isinstance(df['Low'], pd.DataFrame) else df['Low']
    close_prices = df['Close'].values.flatten() if isinstance(df['Close'], pd.DataFrame) else df['Close']
    volume = df['Volume'].values.flatten() if isinstance(df['Volume'], pd.DataFrame) else df['Volume']

    # Perform calculations
    all_time_high = max(high_prices)  # Ensure scalar
    all_time_low = min(low_prices)   # Ensure scalar
    avg_daily_volume = np.mean(volume)  # Ensure scalar

    # Return statistics
    returns = pd.Series(close_prices).pct_change()
    volatility = returns.std() * np.sqrt(252)  # Annualized volatility

    risk_free_rate = 0.01
    excess_returns = returns - risk_free_rate / 252
    sharpe_ratio = np.sqrt(252) * excess_returns.mean() / returns.std()

    rolling_max = pd.Series(close_prices).cummax()
    drawdown = (close_prices - rolling_max) / rolling_max
    max_drawdown = drawdown.min()

    return {
        'all_time_high': all_time_high,
        'all_time_low': all_time_low,
        'avg_daily_volume': avg_daily_volume,
        'volatility': volatility,
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown': max_drawdown
    }


def main():
    st.cache_resource.clear()
    
    # Custom CSS (same as before)
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
    .stProgress 
    .st-bo {
        height: 5px;
    }
    .stDownloadButton {
                display: none;
    } 
    </style>
    """, unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("Settings")
    analysis_type = st.sidebar.selectbox("Select Analysis Type", ["Stocks", "Crypto", "Sentiment"])
    
    
    if analysis_type in ["Stocks", "Crypto"]:
        period = st.sidebar.selectbox("Select Time Period", ['1mo', '3mo', '6mo', '1y', '2y', '5y'])
        is_crypto = analysis_type == "Crypto"
        symbol = st.sidebar.text_input(
            "Enter Symbol", 
            "BTC" if is_crypto else "AAPL"
        ).strip().upper()
        if not symbol:
            st.error("Please enter a valid symbol")
            return
        
        st.markdown(
            f"<h1 class='main-header'>{'🪙' if is_crypto else '📊'} {analysis_type} Analysis Dashboard</h1>", 
            unsafe_allow_html=True
        )
        
        tab1, tab2, tab3, tab4, tab5 = st.tabs(["Overview", "Technical Analysis", "Market Statistics","Metrics", "News"])

        df = fetch_data(symbol, period, is_crypto)
            
        if df is not None and not df.empty:
            df = calculate_technical_indicators(df)
            info = get_asset_info(symbol, is_crypto)
                
            with tab1:
                    # Overview metrics
                col1, col2, col3 = st.columns(3)
                    
                current_price = info.get('currentPrice', 'N/A')
                with col1:
                        if current_price != 'N/A':
                            st.metric("Current Price", f"${float(current_price):,.2f}")
                        else:
                            st.metric("Current Price", "N/A")
                    
                market_cap = info['marketCap']
                with col2:
                        if market_cap != 'N/A':
                            st.metric("Market Cap", f"${float(market_cap):,.0f}")
                        else:
                            st.metric("Market Cap", "N/A")
                    
                with col3:
                        if is_crypto:
                            volume = info['volume24h']
                            if volume != 'N/A':
                                st.metric("24h Volume", f"${float(volume):,.0f}")
                            else:
                                st.metric("24h Volume", "N/A")
                        else:
                            week_change = info['fiftyTwoWeekChange']
                            if week_change != 'N/A':
                                st.metric("52 Week Change", f"{float(week_change):.2%}")
                            else:
                                st.metric("52 Week Change", "N/A")
                    
                    # Price Chart
                chart = create_price_chart(df, symbol)
                if chart:
                    st.plotly_chart(chart, use_container_width=True)
                
            with tab2:
                    # Technical Analysis
                st.subheader("Technical Indicators")
                fig_ma, fig_rsi= create_technical_charts(df)
                    
                st.plotly_chart(fig_ma, use_container_width=True)
                st.plotly_chart(fig_rsi, use_container_width=True)
                
            with tab3:
                    # Market Statistics
                st.subheader("Market Statistics")
                stats = calculate_statistics(df)
                stats_col1, stats_col2 = st.columns(2)
                    
                with stats_col1:
                        st.metric("All-Time High", f"${stats['all_time_high']:,.2f}")
                        st.metric("All-Time Low", f"${stats['all_time_low']:,.2f}")
                        st.metric("Average Daily Volume", f"${stats['avg_daily_volume']:,.0f}")
                    
                with stats_col2:
                        st.metric("Volatility (Annualized)", f"{stats['volatility']:.2%}")
                        st.metric("Sharpe Ratio", f"{stats['sharpe_ratio']:.2f}")
                        st.metric("Maximum Drawdown", f"{stats['max_drawdown']:.2%}")
                
            with tab4:
                if is_crypto:
                        st.subheader("Crypto Metrics")
                        metrics_col1, metrics_col2 = st.columns(2)
                        with metrics_col1:
                            supply_metrics = {
                                "Circulating Supply": info['circulatingSupply'],
                                "Total Supply": info['totalSupply'],
                                "Max Supply": info['maxSupply']
                            }
                            
                            for metric, value in supply_metrics.items():
                                if isinstance(value, (int, float)):
                                    st.metric(metric, f"{value:,.0f}")
                                else:
                                    st.metric(metric, value)
                else:
                    st.subheader("Stock Metrics")
                    metrics_col1, metrics_col2 = st.columns(2)
                        
                    with metrics_col1:
                            st.markdown("### Valuation Metrics")
                            valuation_metrics = {
                                "P/E Ratio": info['trailingPE'],
                                "Forward P/E": info['forwardPE'],
                                "PEG Ratio": info['pegRatio'],
                                "Price to Book": info['priceToBook']
                            }
                            
                            for metric, value in valuation_metrics.items():
                                if isinstance(value, float):
                                    st.metric(metric, f"{value:.2f}")
                                else:
                                    st.metric(metric, value)
                        
                    with metrics_col2:
                            st.markdown("### Growth Metrics")
                            growth_metrics = {
                                "Revenue Growth": info['revenueGrowth'],
                                "Earnings Growth": info['earningsGrowth'],
                                "Profit Margin": info['profitMargins'],
                                "Operating Margin": info['operatingMargins']
                            }
                            
                            for metric, value in growth_metrics.items():
                                if isinstance(value, float):
                                    st.metric(metric, f"{value:.2%}")
                                else:
                                    st.metric(metric, value)
                with tab5:
                    # News
                    st.subheader("Latest News")
                    news = fetch_financial_news(symbol)
                    
                    if news:
                        for article in news:
                            with st.container():
                                st.markdown(f"### [{article['title']}]({article['link']})")
                                st.write(article['description'])
                                st.write(f"Source: {article['source']} | Published: {article['published']}")
                                st.divider()
                    else:
                        st.warning("No news available for this symbol.")

    
    elif analysis_type == "Sentiment":
        st.markdown("<h1 class='main-header'>🎭 Sentiment Analysis Dashboard</h1>", unsafe_allow_html=True)
        tab1 = st.tabs(["Sentiment Analysis"])
        processed_data, metrics, y_test, predictions = get_processed_data()
        if processed_data is not None:
            sentiment_analysis(processed_data, metrics, y_test, predictions)
        else:
            st.error("Error processing sentiment data")


if __name__ == "__main__":
    main()