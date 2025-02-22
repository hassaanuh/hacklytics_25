import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import plotly.graph_objs as go
from plotly.offline import iplot, init_notebook_mode

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import plotly.graph_objs as go
from plotly.offline import plot
import os

def analyze_stock(symbol, path="/Users/hassaanulhaq/.cache/kagglehub/datasets/paultimothymooney/stock-market-data/versions/74/stock_market_data/nyse/csv/"):
    # Read the CSV file
    df = pd.read_csv(f"{path}{symbol}.csv")
    
    # Convert the 'Date' column to datetime objects
    df['Date'] = pd.to_datetime(df['Date'], format='%d-%m-%Y')
    
    # Sort the dataframe by date
    df = df.sort_values(by='Date')
    
    # Create a helper column for grouping
    df['year_month'] = df['Date'].dt.to_period('M')
    
    # Group by year and month, then take the first entry in each group
    df_monthly_start = df.groupby('year_month', as_index=False).first()
    
    # Drop the helper column
    df_monthly_start = df_monthly_start.drop('year_month', axis=1)
    
    # Calculate percent change
    df_monthly_start['Percent Change'] = df_monthly_start['Close'].pct_change()
    
    # Standardize the percent change
    scaler = StandardScaler()
    df_monthly_start['Percent Change'] = scaler.fit_transform(df_monthly_start['Percent Change'].values.reshape(-1,1))
    
    # Fill NaN values
    df_monthly_start['Percent Change'] = df_monthly_start['Percent Change'].fillna(df_monthly_start['Percent Change'].mean())
    
    # Detect anomalies (assuming 3 standard deviations as threshold)
    df_monthly_start['Anomaly'] = np.abs(df_monthly_start['Percent Change']) > 3
    
    # Create the line plot
    trace1 = go.Scatter(
        x=df_monthly_start['Date'], 
        y=df_monthly_start['Percent Change'],
        mode='lines',
        name='Returns'
    )
    
    # Create scatter plot for anomalies
    anomalies = df_monthly_start[df_monthly_start['Anomaly']]
    trace2 = go.Scatter(
        x=anomalies['Date'], 
        y=anomalies['Percent Change'],
        mode='markers',
        name='Anomaly',
        marker=dict(color='red', size=10)
    )
    
    # Combine the traces
    data = [trace1, trace2]
    
    # Define the layout
    layout = go.Layout(
        title=f'Returns Over Time with Anomalies for {symbol}',
        xaxis=dict(title='Date'),
        yaxis=dict(title='Percent Change'),
        hovermode='closest'
    )
    
    # Create the figure
    fig = go.Figure(data=data, layout=layout)
    
    # Save the figure as an HTML file in the current directory
    current_directory = os.getcwd()
    html_file_path = os.path.join(current_directory, f"{symbol}_analysis.html")
    plot(fig, filename=html_file_path, auto_open=False)
    
    print(f"Graph saved as {html_file_path}")

# Main execution
if __name__ == "__main__":
    # Get user input for stock symbol
    symbol = input("Enter the stock symbol (e.g., AAP): ").upper()
    
    try:
        analyze_stock(symbol)
    except FileNotFoundError:
        print(f"Error: CSV file for {symbol} not found. Please check the symbol and file path.")
    except Exception as e:
        print(f"An error occurred: {str(e)}")


