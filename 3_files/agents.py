import openai

# Set OpenAI API Key
OPENAI_API_KEY = ""
client = openai.OpenAI(api_key=OPENAI_API_KEY)


# Convert anomalies into questions for OpenAI processing
def formulate_questions(anomalies, ticker_symbol):
    question = (
            f"On {anomalies}, for the {ticker_symbol} stock price there was an unusual fluctuation. "
            f"Can you investigate possible reasons for this anomaly?"
        )
        
    return question

# Web Research Agent using OpenAI API
def web_research_agent(question):
    completion = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": question}
        ]
    )
    return completion.choices[0].message.content

# Institutional Knowledge Agent using OpenAI API
def institutional_knowledge_agent(question):
  prompt = f"""You are a You are a stock market expert to verify some anomaly data-related questions. 
        Give relevent insights on technical stock indicators and industry indicators which influence to occur the anomaly. 
        Respond a concise summary.
        Question: {question}
        """
  completion = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a financial market expert."},
            {"role": "user", "content": prompt}
        ]
    )
  return completion.choices[0].message.content

# Consolidate responses from all agents into a single report
def consolidate_reports(web_response, institutional_response):
  prompt = f"""
    Your role is to summarise reports from several experts to present to management.
    Each expert tried to answer data-related questions provided below.
    Please concisely summarise the experts' answers.

    Web Research Results: {web_response}
    Institutional Knowledge Results: {institutional_response}
    
    """
  response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a report consolidation expert."},
            {"role": "user", "content": prompt}
        ]
    )
  return response.choices[0].message.content

# Simulate management discussion using another OpenAI API call
def management_discussion(consolidated_report):
    prompt = f"""
    As a panel of management agents, please review and discuss the following consolidated report on S&P 500 Index anomalies. 
    Provide insights from financial market, macroeconomic, and statistical perspectives. Aim to reach a consensus on the report's findings and any necessary actions.
    Lastly what decision you would take at this moment after analyzing the anomolies and resons behind it, you would invest or sell?
    Consolidated Report: {consolidated_report}
    """

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a panel of management experts discussing a financial report."},
            {"role": "user", "content": prompt}
        ]
    )
    return response.choices[0].message.content