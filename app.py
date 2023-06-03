import os
import json
from apikey import apikey

import streamlit as st
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SequentialChain

os.environ['OPENAI_API_KEY'] = apikey

st.title('Actionable Input')
prompt = st.text_input('Your input, such as a question...:')

question_template = PromptTemplate(
    input_variables = ['raw_input'],
    template='Make a question out of the following sentence: "{raw_input}"',
)

#Llms
llm = OpenAI(temperature=0.9)
question_chain = LLMChain(llm=llm, prompt = question_template)


# Show stuff to the screen if the user has entered a prompt
if prompt:
    escaped_prompt = json.dumps(prompt)
    response = question_chain.run(raw_input=escaped_prompt)
    st.write(response)