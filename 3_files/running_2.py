from stock_analysis import run_stock_analysis
from api import run_news_analysis
from datetime import datetime

def get_valid_date(prompt):
    while True:
        date_str = input(prompt)
        try:
            return datetime.strptime(date_str, '%Y-%m-%d').strftime('%Y-%m-%d')
        except ValueError:
            print("Invalid date format. Please use YYYY-MM-DD.")

def main():
    print("Stock Analysis and News Extractor")
    print("=" * 40)

    # Get user input
    csv_path = input("Enter the path to the CSV file: ")
    ticker = input("Enter the stock ticker: ")

    # Run stock analysis
    anomalies_df = run_stock_analysis(csv_path, ticker)

    if anomalies_df.empty:
        print(f"No anomalies found for {ticker}")
    else:
        print(f"Number of anomalies found: {len(anomalies_df)}")
        print("\nAnomalies detected:")
        print(anomalies_df)

    # Get user input for date range
    print("\nPlease enter the date range for news analysis:")
    start_date = get_valid_date("Start Date (YYYY-MM-DD): ")
    end_date = get_valid_date("End Date (YYYY-MM-DD): ")

    # Run news analysis
    print(f"\nFetching news summary for {ticker} from {start_date} to {end_date}...")
    news_summary = run_news_analysis(start_date, end_date, ticker)

    print("\nRelated News Summary:")
    print("=" * 40)
    print(news_summary)

if __name__ == "__main__":
    main()
