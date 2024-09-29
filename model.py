import os
import streamlit as st
from langchain.chains import LLMChain
from langchain import PromptTemplate
from langchain_huggingface import HuggingFaceEndpoint

sec_key = "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"
repo_id = "mistralai/Mistral-7B-Instruct-v0.3"

llm_gen = HuggingFaceEndpoint(
    repo_id=repo_id,
    max_length=512,
    temperature=0.7,
    token=sec_key
)

q_gen_template = '''
Given the context below, generate answer:

Context: {context}

Question:
'''
prompt_gen = PromptTemplate(
    input_variables=['context'],
    template=q_gen_template
)

credit_risk_template = '''
Assess the credit risk for a customer based on the following data:

Customer Data: {data}

Credit Risk:
'''
prompt_risk = PromptTemplate(
    input_variables=['data'],
    template=credit_risk_template
)

sentiment_analysis_template = '''
Analyze the sentiment of the following financial text and generate insights:

Text: {text}

Sentiment and Insights:
'''
prompt_sentiment = PromptTemplate(
    input_variables=['text'],
    template=sentiment_analysis_template
)

# Personalized Financial Advice Prompt
financial_advice_template = '''
Provide personalized financial advice based on the following user situation:

Situation: {situation}

Advice:
'''
prompt_advice = PromptTemplate(
    input_variables=['situation'],
    template=financial_advice_template
)


# Financial Report Generation Prompt
report_generation_template = '''
Generate a financial report based on the following data:

Data: {data}

Report:
'''
prompt_report = PromptTemplate(
    input_variables=['data'],
    template=report_generation_template
)

# Create Streamlit App UI
st.title("Financial Generative AI Assistant")

st.sidebar.title("Choose a Financial Task")
task = st.sidebar.selectbox(
    "Task",
    (
        "Automatic Financial Question Answering",
        "Financial Sentiment Analysis with Text Generation",
        "Generative AI for Personalized Financial Advice",
        "AI-Driven Financial Report Generation",
        "Credit Risk Assessment"
    ),
)

if task == "Automatic Financial Question Answering":
    st.header("Automatic Financial Question Answering")
    context = st.text_area("Enter context for question generation:")
    if st.button("Generate Answer"):
        if context:
            # Create an LLMChain for question generation
            q_chain = LLMChain(llm=llm_gen, prompt=prompt_gen)
            response = q_chain.run({"context": context})
            st.write("### Generated Answer:")
            st.write(response)
        else:
            st.write("Please enter some context.")

elif task == "Financial Sentiment Analysis with Text Generation":
    st.header("Financial Sentiment Analysis with Text Generation")
    text = st.text_area("Enter financial text for sentiment analysis:")
    if st.button("Analyze Sentiment"):
        if text:
            # Create an LLMChain for sentiment analysis and insight generation
            sentiment_chain = LLMChain(llm=llm_gen, prompt=prompt_sentiment)
            response = sentiment_chain.run({"text": text})
            st.write("### Sentiment and Insights:")
            st.write(response)
        else:
            st.write("Please enter some financial text.")

elif task == "Credit Risk Assessment":
    st.header("Credit Risk Assessment")
    data = st.text_area("Enter customer data for credit risk assessment:")
    if st.button("Assess Credit Risk"):
        if data:
            # Create an LLMChain for credit risk assessment
            risk_chain = LLMChain(llm=llm_gen, prompt=prompt_risk)
            response = risk_chain.run({"data": data})
            st.write("### Credit Risk Assessment:")
            st.write(response)
        else:
            st.write("Please provide customer data.")            

elif task == "Generative AI for Personalized Financial Advice":
    st.header("Generative AI for Personalized Financial Advice")
    situation = st.text_area("Describe your financial situation for personalized advice:")
    if st.button("Get Advice"):
        if situation:
            # Create an LLMChain for personalized financial advice
            advice_chain = LLMChain(llm=llm_gen, prompt=prompt_advice)
            response = advice_chain.run({"situation": situation})
            st.write("### Financial Advice:")
            st.write(response)
        else:
            st.write("Please describe your financial situation.")

elif task == "AI-Driven Financial Report Generation":
    st.header("AI-Driven Financial Report Generation")
    data = st.text_area("Enter data for generating a financial report:")
    if st.button("Generate Report"):
        if data:
            # Create an LLMChain for report generation
            report_chain = LLMChain(llm=llm_gen, prompt=prompt_report)
            response = report_chain.run({"data": data})
            st.write("### Generated Financial Report:")
            st.write(response)
        else:
            st.write("Please provide input for the financial report.")

# Footer and contact info
st.sidebar.info("Developed by Rachit Ranjan.")
