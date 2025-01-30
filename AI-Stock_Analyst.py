# stock_analysis_app.py
import yfinance as yf
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from langchain.chains import LLMChain
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from googlesearch import search
import numpy as np
import os
from dotenv import load_dotenv

load_dotenv()

# Configuration
st.set_page_config(page_title="AI Stock Analyst", layout="wide")

def get_stock_data(ticker):
    """Fetch stock data from Yahoo Finance"""
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period="1y")
        return hist
    except Exception as e:
        st.error(f"Error fetching data for {ticker}: {str(e)}")
        return None

def calculate_technical_indicators(data):
    """Calculate technical indicators"""
    data['50_SMA'] = data['Close'].rolling(window=50).mean()
    data['200_SMA'] = data['Close'].rolling(window=200).mean()
    
    # Calculate RSI
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    data['RSI'] = 100 - (100 / (1 + rs))
    
    return data

def plot_stock_data(data, ticker):
    """Create interactive visualization with technical indicators"""
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                       vertical_spacing=0.1, 
                       row_heights=[0.7, 0.3],
                       subplot_titles=(f'{ticker} Price Action', 'RSI'))

    # Candlestick chart
    fig.add_trace(go.Candlestick(x=data.index,
                                open=data['Open'],
                                high=data['High'],
                                low=data['Low'],
                                close=data['Close'],
                                name='Price'), row=1, col=1)

    # Moving averages
    fig.add_trace(go.Scatter(x=data.index, y=data['50_SMA'],
                            line=dict(color='orange', width=2),
                            name='50 SMA'), row=1, col=1)
    
    fig.add_trace(go.Scatter(x=data.index, y=data['200_SMA'],
                            line=dict(color='blue', width=2),
                            name='200 SMA'), row=1, col=1)

    # RSI
    fig.add_trace(go.Scatter(x=data.index, y=data['RSI'],
                            line=dict(color='purple', width=2),
                            name='RSI'), row=2, col=1)

    # RSI markers
    fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)

    fig.update_layout(
        height=800,
        showlegend=True,
        xaxis_rangeslider_visible=False,
        template='plotly_dark'
    )
    
    return fig

def get_recent_news(ticker):
    """Fetch recent news using Google Search"""

    try:
            query = f"{ticker} stock news latest updates"
            # Fix the search parameters
            return []
    
             
    except Exception as e:
            st.warning(f"Could not fetch news: {str(e)}")
            return []

def generate_technical_analysis(data, ticker):
    """Generate detailed technical analysis from graph data"""
    latest = data.iloc[-1]
    prev_close = data['Close'].iloc[-2]
    
    # Price movement analysis
    price_change_1d = ((latest['Close'] - prev_close) / prev_close) * 100
    price_change_1m = ((latest['Close'] - data['Close'].iloc[-22]) / data['Close'].iloc[-22]) * 100
    price_change_3m = ((latest['Close'] - data['Close'].iloc[-66]) / data['Close'].iloc[-66]) * 100
    
    # Moving average analysis
    ma_signal = ""
    if latest['50_SMA'] > latest['200_SMA']:
        ma_signal = "Golden Cross (50-day SMA > 200-day SMA) - Long-term bullish signal"
    else:
        ma_signal = "Death Cross (50-day SMA < 200-day SMA) - Long-term bearish signal"
        
    # RSI analysis
    rsi_trend = "rising" if latest['RSI'] > data['RSI'].iloc[-5] else "falling"
    rsi_strength = ""
    if latest['RSI'] > 70:
        rsi_strength = "Overbought - Potential pullback expected"
    elif latest['RSI'] < 30:
        rsi_strength = "Oversold - Potential rebound possible"
    else:
        rsi_strength = "Neutral territory"
    
    # Volume analysis
    volume_change = ((latest['Volume'] - data['Volume'].rolling(14).mean().iloc[-1]) 
                    / data['Volume'].rolling(14).mean().iloc[-1]) * 100
    
    # Support/resistance levels
    support = data['Low'].rolling(20).min().iloc[-1]
    resistance = data['High'].rolling(20).max().iloc[-1]
    
    analysis = f"""
    **Technical Breakdown for {ticker}**
    
    📈 **Price Action**
    - Current Price: ${latest['Close']:.2f}
    - 24h Change: {price_change_1d:+.2f}%
    - 1M Return: {price_change_1m:+.2f}%
    - 3M Return: {price_change_3m:+.2f}%
    
    📊 **Moving Averages**
    - 50-Day SMA: ${latest['50_SMA']:.2f} ({'Above' if latest['Close'] > latest['50_SMA'] else 'Below'} current price)
    - 200-Day SMA: ${latest['200_SMA']:.2f} ({'Above' if latest['Close'] > latest['200_SMA'] else 'Below'} current price)
    - Trend Signal: {ma_signal}
    
    🔄 **Momentum Indicators**
    - RSI (14-day): {latest['RSI']:.1f} ({rsi_strength})
    - RSI Trend: {rsi_trend} over past 5 sessions
    - Support Level: ${support:.2f}
    - Resistance Level: ${resistance:.2f}
    
    💹 **Volume Insights**
    - Recent Volume: {latest['Volume']:,} shares
    - Volume Change vs 14D Avg: {volume_change:+.1f}%
    """
    
    return analysis

def analyze_with_ai(ticker, data, news):
    """Generate investment recommendation using AI"""

      # GROQ Integration with DeepSeek
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    key = GROQ_API_KEY
    llm = ChatGroq(groq_api_key=key , model_name ="deepseek-r1-distill-llama-70b") 
    
    
    current_price = data['Close'].iloc[-1]
    sma50 = data['50_SMA'].iloc[-1]
    sma200 = data['200_SMA'].iloc[-1]
    rsi = data['RSI'].iloc[-1]
    
    template = """As a senior financial analyst, analyze {ticker} stock considering:

    Technical Analysis:
    - Current Price: ${price:.2f}
    - 50-Day SMA: ${sma50:.2f} ({sma50_status})
    - 200-Day SMA: ${sma200:.2f} ({sma200_status})
    - RSI: {rsi:.1f} ({rsi_status})
    
    Recent News:
    {news}

    Fundamental Considerations:
    - Market trends in the sector
    - Company financial health
    - Economic indicators
    - Risk factors

    Provide detailed analysis and final recommendation. Conclude with either:
    "Recommendation: [Invest/Do Not Invest] - [Brief Reason]" 
    Format the recommendation in bold at the end."""

    prompt = PromptTemplate(
        input_variables=["ticker", "price", "sma50", "sma200", "rsi", "news"],
        template=template
    )
    
    chain = LLMChain(llm=llm, prompt=prompt)
    
    return chain.run({
        "ticker": ticker,
        "price": current_price,
        "sma50": sma50,
        "sma50_status": "Bullish" if current_price > sma50 else "Bearish",
        "sma200": sma200,
        "sma200_status": "Bullish" if current_price > sma200 else "Bearish",
        "rsi": rsi,
        "rsi_status": "Overbought" if rsi > 70 else "Oversold" if rsi < 30 else "Neutral",
        "news": "\n- ".join(news[:3])  # Top 3 news items
    })

def main():
    st.title("🤖 AI-Powered Stock Analysis Platform")
    
    col1, col2 = st.columns([1, 3])
    with col1:
        ticker = st.text_input("Enter Stock Ticker:", "TSLA").upper()
        analysis_period = st.selectbox("Analysis Period", ["1y", "6mo", "3mo", "1mo","present"])
    
    # Modify the main() function's button click section:
    if st.button("Analyze Stock"):
        with st.spinner("Crunching numbers..."):
            data = get_stock_data(ticker)
            
            if data is None or data.empty:
                st.error("Invalid ticker or no data available")
                return
                
            data = calculate_technical_indicators(data)
            news = get_recent_news(ticker)
            
            # Display stock chart
            st.subheader(f"Technical Analysis - {ticker}")
            fig = plot_stock_data(data, ticker)
            st.plotly_chart(fig, use_container_width=True)
            
            # New technical analysis section
            st.subheader("📉 Detailed Technical Breakdown")
            technical_report = generate_technical_analysis(data, ticker)
            st.markdown(technical_report)
            
            # AI Analysis
            st.subheader("🤖 AI Investment Analysis")
            analysis = analyze_with_ai(ticker, data, news)
            
            # Display recommendation with emphasis
            if "Recommendation: Invest" in analysis:
                st.success(analysis.split("Recommendation:")[-1])
            elif "Recommendation: Do Not Invest" in analysis:
                st.error(analysis.split("Recommendation:")[-1])
            else:
                st.info(analysis)
            
            # Show raw data
            with st.expander("Show Raw Data"):
                st.dataframe(data.tail(10))
if __name__ == "__main__":
    main()