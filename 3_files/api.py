from flask import Flask, request, jsonify, send_file
import os
from openai import OpenAI
from datetime import datetime, timedelta
from stock_analysis import analyze_stock # Import the stock analysis functions
import pandas as pd

app = Flask(__name__)

# OpenAI API Key (Replace with your actual OpenAI API key)
OPENAI_API_KEY ="Enter API KEy"

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

def run_news_analysis(start_date, end_date, keyword):
    news_summary = get_news_summary(start_date, end_date, keyword)
    return news_summary


