import os
from dotenv import load_dotenv
# from langchain_ollama import OllamaLLM  # Updated import
from langchain_ollama import ChatOllama
import streamlit as st
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()
os.environ['LANGCHAIN_API_KEY'] = os.getenv('LANGCHAIN_API_KEY')
os.environ['LANGCHAIN_PROJECT'] = os.getenv('LANGCHAIN_PROJECT')

# Define prompt template
prompt = ChatPromptTemplate.from_template(
    template="You are a helpful assistant. Please respond to the question asked.\nQuestion: {question}"
)

# Streamlit framework
st.title("Langchain demo with gemma:2b")
input_text = st.text_input("What question do you have in mind?")

# Use the updated OllamaLLM class
try:
    llm = ChatOllama(
    model = "gemma:2b",
    temperature = 0.8,
    num_predict = 256,
    # other params ...
    )
    output_parser = StrOutputParser()
    chain = prompt | llm | output_parser

    if input_text:
        response = chain.invoke({"question": input_text})
        st.write(response)

except Exception as e:
    st.error(f"An error occurred: {e}")
