from openai import OpenAI

from datetime import datetime, timedelta

# OpenAI API Key (Replace with your actual OpenAI API key)
OPENAI_API_KEY = "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"

# Set the API Key for OpenAI

client = OpenAI(api_key=OPENAI_API_KEY)


def get_news_summary(start_date, end_date, keyword):
    """
    Fetches finance news summary using OpenAI's ChatGPT API.
    """
    # Constructing the prompt
    prompt = (
        f"Summarize the top finance news about '{keyword}' "
        f"from {start_date} to {end_date}. Include key events, trends, and impacts, and any major stock market movements and corporate actions"
    )

    # Making the request to OpenAI's ChatGPT API using the latest syntax
    response = client.chat.completions.create(model="gpt-4o",
    messages=[
        {"role": "system", "content": "You are a financial news analyst."},
        {"role": "user", "content": prompt}
    ],
    max_tokens=800,
    temperature=0.5)

    # Extracting the content of the response
    news_summary = response.choices[0].message.content
    return news_summary

def main():
    """
    Main function to run the news extraction program.
    """
    print("Finance News Extractor (Powered by OpenAI)")
    print("=" * 40)
    print("Enter the date range and keyword to extract top finance news.")

    # User input for date range and keyword (No strict date checking)
    start_date = input("Start Date (YYYY-MM-DD): ")
    end_date = input("End Date (YYYY-MM-DD): ")
    keyword = input("Keyword: ")

    # Fetching the news summary using OpenAI's API
    news_summary = get_news_summary(start_date, end_date, keyword)

    # Displaying the extracted news
    print("\nTop Finance News Summary:")
    print("=" * 40)
    print(news_summary)

if __name__ == "__main__":
    main()
