import os
import pandas as pd
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent

load_dotenv(dotenv_path="../.env")

DATA_PATH = "../data/titanic.csv"

def process_query(question: str) -> dict:
    try:
        df = pd.read_csv(DATA_PATH)
        
        llm = ChatGroq(
            temperature=0,
            model_name="llama-3.3-70b-versatile",
            api_key=os.getenv("GROQ_API_KEY")
        )

        agent = create_pandas_dataframe_agent(
            llm, 
            df, 
            verbose=True,
            allow_dangerous_code=True,
            handle_parsing_errors=True
        )
        
        prompt = f"""
        Question: {question}

        STRICT INSTRUCTIONS FOR YOUR FINAL ANSWER:
        1. You MUST begin your response with "Final Answer: ".
        2. Provide a detailed and conversational response. Instead of just a number, explain what the data shows in 2-3 sentences.
        3. If the user asks for a visualization, explain what the chart represents before providing the ```python code block.
        4. For any histograms requested, always include a title, x-axis label, y-axis label, and black edge colors for the bars.
        """
        
        response = agent.invoke(prompt)
        return {"answer": response["output"]}
        
    except Exception as e:
        return {"answer": f"Error processing query: {str(e)}"}