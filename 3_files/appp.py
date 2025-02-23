import streamlit as st
import pandas as pd
import plotly.graph_objs as go
import yfinance as yf
from datetime import datetime, timedelta
import openai
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as torch_data
import os
from scipy.stats import zscore
from agents import formulate_questions, web_research_agent, institutional_knowledge_agent, consolidate_reports, management_discussion

# Set Streamlit Page Configurations
st.set_page_config(page_title="Trading Learning through Anomalies", layout="wide")

st.title("📈 Trading Learning Through Anomalies")

# Define stock tickers
stocks = ["AMZN", "AAPL", "WMT", "MSFT", "TSLA", "WFC", "NVDA"]

def fetch_stock_data(ticker):
    stock = yf.Ticker(ticker)
    df = stock.history(period="5y")
    return df.reset_index()

def analyze_stock(df, selected_ticker):
    df['Date'] = pd.to_datetime(df['Date'])
    df['Close'] = df['Close'].astype(float)
    lookback = 5
    timeseries = df[['Close']].values.astype('float32')
    X, y = [], []
    for i in range(len(timeseries)-lookback):
        X.append(timeseries[i:i+lookback])
        y.append(timeseries[i+1:i+lookback+1])
    X, y = torch.tensor(X), torch.tensor(y)

    class StockModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.lstm = nn.LSTM(input_size=1, hidden_size=50, num_layers=1, batch_first=True)
            self.linear = nn.Linear(50, 1)
        def forward(self, x):
            x, _ = self.lstm(x)
            return self.linear(x)

    model = StockModel()
    optimizer = optim.Adam(model.parameters())
    loss_fn = nn.MSELoss()
    loader = torch_data.DataLoader(torch_data.TensorDataset(X, y), shuffle=True, batch_size=8)
    for epoch in range(50):
        for batch_X, batch_y in loader:
            optimizer.zero_grad()
            loss = loss_fn(model(batch_X), batch_y)
            loss.backward()
            optimizer.step()
    model.eval()
    with torch.no_grad():
        pred_series = torch.full_like(torch.tensor(timeseries), float('nan'))
        pred_series[lookback:] = model(X)[:, -1, :]
    error = abs(df['Close'].values - pred_series.numpy().flatten())
    anomalies = df.loc[error > 6, ['Date', 'Close']]
    return anomalies

def detect_anomalies(data, threshold=2):
    data['Returns'] = data['Close'].pct_change()
    data['Z-Score'] = zscore(data['Returns'], nan_policy='omit')
    return data[abs(data['Z-Score']) > threshold][['Date', 'Close']].reset_index(drop=True)

# Layout with sidebar selection
col1, col2 = st.columns([1, 2])
with col1:
    selected_ticker = st.selectbox("**Select Stock**", stocks)
    algos = ["Standard Scalar - Z-Score", "Deep Learning - LSTM"]
    selected_algo = st.selectbox("**Select Anomaly Detection Algorithm**", algos)
    if st.button("Analyze Stock"):
        df = fetch_stock_data(selected_ticker)
        st.session_state['df'] = df
        st.session_state['anomalies'] = detect_anomalies(df) if selected_algo == "Z-Score" else analyze_stock(df, selected_ticker)

    if 'anomalies' in st.session_state and not st.session_state['anomalies'].empty:
        st.session_state['selected_anomaly'] = st.selectbox("**Select an anomaly date**", st.session_state['anomalies']['Date'].astype(str))
        if st.button("Analyze Anomaly"):
            st.session_state['question'] = formulate_questions(st.session_state['selected_anomaly'], selected_ticker)
            st.session_state['web_response'] = web_research_agent(st.session_state['question'])
            st.session_state['institutional_response'] = institutional_knowledge_agent(st.session_state['question'])
            st.session_state['consolidated_report'] = consolidate_reports(st.session_state['web_response'], st.session_state['institutional_response'])
            st.session_state['management_response'] = management_discussion(st.session_state['consolidated_report'])

with col2:
    if 'df' in st.session_state and 'anomalies' in st.session_state:
        st.subheader("📊 Stock Price Trend & Anomalies")
        trace1 = go.Scatter(x=st.session_state['df']['Date'], y=st.session_state['df']['Close'], mode='lines', name='Closing Price')
        trace2 = go.Scatter(x=st.session_state['anomalies']['Date'], y=st.session_state['anomalies']['Close'], mode='markers', name='Anomaly', marker=dict(color='red', size=8))
        fig = go.Figure(data=[trace1, trace2])
        fig.update_layout(title=f"Stock Price Trend for {selected_ticker}", xaxis_title="Date", yaxis_title="Price", hovermode='closest')
        st.plotly_chart(fig)

    if 'management_response' in st.session_state:
        st.markdown("---")
        st.subheader("🧐 AI-Powered Anomaly Analysis")
        st.markdown("### 🔍 Question to Investigate the Anomaly")
        st.info(st.session_state['question'])
        st.markdown("### 🌍 Web Research Findings")
        st.success(st.session_state['web_response'])
        st.markdown("### 🏛️ Institutional Knowledge")
        st.success(st.session_state['institutional_response'])
        st.markdown("### 📜 Consolidated Report")
        st.warning(st.session_state['consolidated_report'])
        st.markdown("### 🏢 Management Discussion")
        st.error(st.session_state['management_response'])
