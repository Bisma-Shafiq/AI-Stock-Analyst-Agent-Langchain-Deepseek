# AI Stock Analysis Platform

Welcome to the AI Stock Analysis Platform! This is a Streamlit-based web application designed to provide AI-powered stock analysis, including real-time technical indicators, stock data visualizations, and AI-generated investment recommendations. 

## Features

- **Real-Time Stock Data**: Fetches historical stock data for the past year using Yahoo Finance API.
- **Technical Indicators**: Calculates and visualizes key indicators such as:
  - 50-Day and 200-Day Simple Moving Averages (SMA)
  - Relative Strength Index (RSI)
  - Price action and volume analysis
- **Interactive Visualizations**: Uses Plotly to create interactive candlestick charts, moving averages, and RSI indicators.
- **AI-Powered Analysis**: Integration with AI models to provide investment recommendations based on technical analysis and recent news.
- **News Fetching**: Displays the latest news about the stock to enhance investment decision-making.
- **Stock Recommendation**: Provides investment recommendations based on technical analysis and AI insights.

## Technologies Used

- **Python**: Core programming language.
- **Streamlit**: For building the interactive web application.
- **Yahoo Finance API (`yfinance`)**: To fetch stock data.
- **Plotly**: For interactive data visualization.
- **LangChain**: For AI integration and generating investment insights.
- **Google Search**: For fetching the latest news about the stock.
- **dotenv**: To load environment variables securely (e.g., API keys).
- **Groq**: For advanced AI-powered stock analysis via the ChatGroq model.
- **DEEPSEEK**: For advanced AI-powered LLM model.
- 
## Setup

To run the application locally, follow these steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/AI-Stock-Analysis-Platform.git
