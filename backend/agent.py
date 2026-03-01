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
            agent_type="tool-calling", # This avoids the Thought/Action parsing errors
            allow_dangerous_code=True,
            handle_parsing_errors=True
        )
        
        prompt = f"""
        User Question: {question}

        Instructions:
        1. Provide a detailed 2-3 sentence explanation of your findings.
        2. If a visualization is requested, explain the data first, then provide the ```python code.
        3. Histograms must include a title, x-axis label, y-axis label, and black edge colors.
        """
        
        response = agent.invoke(prompt)
        return {"answer": response["output"]}
        
    except Exception as e:
        return {"answer": f"Error processing query: {str(e)}"}