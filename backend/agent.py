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
        )
        
        prompt = f"""
        Question: {question}

        STRICT INSTRUCTIONS:
        1. You MUST begin your response with the phrase "Final Answer: " and give a little detailed explanation around this final answer.
        2. If the user asks for a fact or statistic, give a text-only response after "Final Answer: ".
        3. If the user asks for a chart or visualization, provide the Python code after "Final Answer: " wrapped in ```python blocks.
        4. For histograms, include a title, x-axis label, y-axis label, and black edge colors.
        """
        
        response = agent.invoke(prompt)
        return {"answer": response["output"]}
        
    except Exception as e:
        return {"answer": f"Error processing query: {str(e)}"}