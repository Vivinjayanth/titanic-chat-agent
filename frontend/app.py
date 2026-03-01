import streamlit as st
import requests
import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

st.title("Titanic Dataset Chat Agent")
st.write("Ask questions about the Titanic dataset in plain English.")

# Updated to your live Render backend
API_URL = "https://titanic-chat-agent-20gw.onrender.com/api/chat"
# Fetch directly from URL for frontend visualizations
DATA_URL = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if message.get("content"):
            st.markdown(message["content"])
        if "plot_code" in message and message["plot_code"]:
            try:
                df = pd.read_csv(DATA_URL)
                local_vars = {"df": df, "plt": plt, "sns": sns}
                exec(message["plot_code"], globals(), local_vars)
                st.pyplot(plt.gcf())
                plt.clf()
            except Exception:
                pass

if prompt := st.chat_input("E.g., What percentage of passengers were male?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        try:
            response = requests.post(API_URL, json={"question": prompt}).json()
            answer = response.get("answer", "")
            
            code_match = re.search(r'```python\n(.*?)\n```', answer, re.DOTALL)
            plot_code = None
            text_content = answer
            
            if code_match:
                plot_code = code_match.group(1)
                text_content = answer.replace(f"```python\n{plot_code}\n```", "").strip()
                
            if text_content:
                st.markdown(text_content)
                
            if plot_code:
                try:
                    df = pd.read_csv(DATA_URL)
                    local_vars = {"df": df, "plt": plt, "sns": sns}
                    exec(plot_code, globals(), local_vars)
                    st.pyplot(plt.gcf())
                    plt.clf()
                except Exception as e:
                    st.error(f"Could not render plot: {e}")
                    
            st.session_state.messages.append({
                "role": "assistant", 
                "content": text_content, 
                "plot_code": plot_code
            })
            
        except requests.exceptions.ConnectionError:
            st.error("Connection failed. Check if the backend is live.")