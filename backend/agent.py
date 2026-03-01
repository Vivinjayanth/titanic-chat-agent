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
            handle_parsing_errors=True,
            suffix="Begin! You must always start your final response with 'Final Answer: ' and provide a detailed 2-3 sentence explanation."
        )
        
        prompt = f"""
        User Question: {question}

        Follow these rules strictly:
        - Your response must start with 'Final Answer: '.
        - Provide a conversational, detailed 2-3 sentence explanation of the data found.
        - If a chart is requested, include the detailed explanation, then the ```python code block.
        - Histograms must have a title, x-axis label, y-axis label, and black edge colors.
        """
        
        response = agent.invoke(prompt)
        return {"answer": response["output"]}
        
    except Exception as e:
        return {"answer": f"Error processing query: {str(e)}"}