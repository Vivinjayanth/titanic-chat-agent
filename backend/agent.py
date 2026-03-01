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
        Answer the following question: {question}

        CRITICAL INSTRUCTIONS FOR YOUR FINAL ANSWER:
        1. If the question does NOT explicitly ask for a plot, chart, or visualization, provide a text-only Final Answer. Do NOT output any Python code.
        2. ONLY IF the question explicitly asks for a visualization, provide the Python code in your Final Answer wrapped strictly in ```python and ``` blocks. Use matplotlib.pyplot as plt. Do not include plt.show().
        3. If the requested visualization is a histogram, you MUST include a title, x-axis label, y-axis label, and set the edge color to black (e.g., edgecolor='black').
        """
        
        response = agent.invoke(prompt)
        return {"answer": response["output"]}
        
    except Exception as e:
        return {"answer": f"Error processing query: {str(e)}"}