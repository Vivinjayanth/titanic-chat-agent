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
        User Question: {question}

        YOU MUST FOLLOW THESE RULES:
        1. Start your response exactly with "Final Answer: ". 
        2. Provide a detailed, 2-3 sentence explanation of the data.
        3. If the user asks for a chart, include the explanation first, then the ```python code block.
        4. For histograms, include a title, x-axis label, y-axis label, and black edge colors.

        Example of a correct response:
        Final Answer: There were 577 male passengers on the Titanic. This indicates that males made up approximately 65% of the total passengers on the ship.
        """
        
        response = agent.invoke(prompt)
        return {"answer": response["output"]}
        
    except Exception as e:
        return {"answer": f"Error processing query: {str(e)}"}