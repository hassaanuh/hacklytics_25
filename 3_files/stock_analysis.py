import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as torch_data
import plotly.graph_objs as go
from flask import Flask, request, jsonify, send_file
import os

app = Flask(__name__)

def analyze_stock(df_param, selected_ticker):
    print(f"Analyzing stock: {selected_ticker}")
    
    # Convert date column to datetime
    df_param['date'] = pd.to_datetime(df_param['date'], format='%Y%m%d')

    def detect_and_adjust_splits_for_all_stocks(df):
        # ... [function implementation remains unchanged] ...
        return df

    # Filter the dataframe based on the selected stock ticker
    filtered_df = df_param[df_param['stock_ticker'] == selected_ticker]
    filtered_df = detect_and_adjust_splits_for_all_stocks(filtered_df)
    
    if filtered_df.empty:
        print(f"No data found for ticker: {selected_ticker}")
        return None

    filtered_df['prc'] = filtered_df['prc'].astype(float)

    # Prepare data for LSTM
    lookback = 5
    timeseries = filtered_df[["prc"]].values.astype('float32')
    X, y = [], []
    for i in range(len(timeseries)-lookback):
        feature = timeseries[i:i+lookback]
        target = timeseries[i+1:i+lookback+1]
        X.append(feature)
        y.append(target)
    
    X = torch.tensor(X)
    y = torch.tensor(y)

    print(f"Data shape - X: {X.shape}, y: {y.shape}")

    # Define and train the model
    class StockModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.lstm = nn.LSTM(input_size=1, hidden_size=50, num_layers=1, batch_first=True)
            self.linear = nn.Linear(50, 1)
        def forward(self, x):
            x, _ = self.lstm(x)
            x = self.linear(x)
            return x

    model = StockModel()
    optimizer = optim.Adam(model.parameters())
    loss_fn = nn.MSELoss()
    loader = torch_data.DataLoader(torch_data.TensorDataset(X, y), shuffle=True, batch_size=8)

    print("Training model...")
    for epoch in range(100):  # Reduced epochs for faster execution
        for batch_X, batch_y in loader:
            optimizer.zero_grad()
            y_pred = model(batch_X)
            loss = loss_fn(y_pred, batch_y)
            loss.backward()
            optimizer.step()
        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

    # Make predictions
    model.eval()
    with torch.no_grad():
        pred_series = np.ones_like(timeseries) * np.nan
        pred_series[lookback:] = model(X)[:, -1, :]

    # Calculate error and detect anomalies
    error = abs(filtered_df['prc'].values - pred_series.flatten())
    error_series = pd.Series(error, index=filtered_df['date'])
    price_series = pd.Series(filtered_df['prc'].values, index=filtered_df['date'])

    threshold = 6  # Adjust this value as needed
    anomalies_filter = error_series > threshold
    anomalies = price_series[anomalies_filter]

    print(f"Number of anomalies detected: {len(anomalies)}")

    # Create the plot
    trace1 = go.Scatter(
        x=filtered_df['date'], 
        y=filtered_df['prc'],
        mode='lines',
        name='Closing Price'
    )

    trace2 = go.Scatter(
        x=anomalies.index, 
        y=anomalies.values,
        mode='markers',
        name='Anomaly',
        marker=dict(color='red', size=5)
    )

    plot_data = [trace1, trace2]

    layout = go.Layout(
        title=f'Closing Price Over Time with Anomalies for {selected_ticker}',
        xaxis=dict(
            rangeselector=dict(
                buttons=list([
                    dict(count=6, label="6m", step="month", stepmode="backward"),
                    dict(count=1, label="YTD", step="year", stepmode="todate"),
                    dict(count=1, label="1y", step="year", stepmode="backward"),
                    dict(count=0, label="This Year", step="year", stepmode="todate"),
                    dict(step="all")
                ])
            ),
            rangeslider=dict(visible=True),
            type="date"
        ),
        yaxis=dict(title='Price'),
        hovermode='closest'
    )

    fig = go.Figure(data=plot_data, layout=layout)
    
    # Export the graph as an HTML file
    output_file = f"{selected_ticker}_stock_analysis.html"
    fig.write_html(output_file)
    print(f"Analysis complete. Graph saved to {output_file}")
    return output_file

def create_anomalies_table(anomalies):
    anomalies_df = anomalies.reset_index()
    anomalies_df.columns = ['Date', 'Price']
    anomalies_df['Date'] = anomalies_df['Date'].dt.strftime('%Y-%m-%d')
    return anomalies_df

def create_anomalies_table(anomalies):
    if isinstance(anomalies, pd.Series) and not anomalies.empty:
        anomalies_df = anomalies.reset_index()
        anomalies_df.columns = ['Date', 'Price']
        anomalies_df['Date'] = anomalies_df['Date'].dt.strftime('%Y-%m-%d')
        return anomalies_df
    else:
        return pd.DataFrame(columns=['Date', 'Price'])


def run_stock_analysis(csv_path, ticker):
    df_param = pd.read_csv(csv_path)
    anomalies = analyze_stock(df_param, ticker)
    return create_anomalies_table(anomalies)

# Load the data once when the server starts
# df_param = pd.read_csv('/Users/hassaanulhaq/Library/Mobile Documents/com~apple~CloudDocs/Hacklytics/hacklytics_25/click_input/hackathon_sample_v2.csv')

# @app.route('/analyze', methods=['POST'])
# def analyze():
#     data = request.json
#     ticker = data.get('ticker')
#     if not ticker:
#         return jsonify({"error": "No ticker provided"}), 400
    
#     output_file = analyze_stock(df_param, ticker)
#     if output_file:
#         return jsonify({"message": "Analysis complete", "file": output_file})
#     else:
#         return jsonify({"error": "Analysis failed"}), 500

# @app.route('/get_graph/<ticker>')
# def get_graph(ticker):
#     file_path = f"{ticker}_stock_analysis.html"
#     if os.path.exists(file_path):
#         return send_file(file_path)
#     else:
#         return jsonify({"error": "Graph not found"}), 404

# if __name__ == "__main__":
#     app.run(debug=True)
