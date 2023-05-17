import os
from apikey import apikey

import streamlit as st
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SequentialChain

os.environ['OPENAI_API_KEY'] = apikey

st.title('Problem')
prompt = st.text_input('What are you trying to solve?:')

st.title('Challenge #1')
prompt = st.text_input('What is one thing that makes solving this difficult:')

st.title('Challenge #2')
prompt = st.text_input('What is a second thing that makes solving this difficult:')

st.title('Challenge #3')
prompt = st.text_input('What is a third thing that makes solving this difficult:')

#Prompt templates
title_template = PromptTemplate(
    input_variables = ['topic'],
    template = 'write me a youtube video title about {topic}'
)

#Prompt templates
script_template = PromptTemplate(
    input_variables = ['title'],
    template = 'write me a youtube video script based on this title Title: {title}'
)

#Llms
llm = OpenAI(temperature=0.9)
title_chain = LLMChain(llm=llm, prompt=title_template, verbose=True, output_key='title')
script_chain = LLMChain(llm=llm, prompt=script_template, verbose=True, output_key='script')
sequential_chain = SequentialChain(chains=[title_chain, script_chain], input_variables=['topic'], output_variables=['title', 'script'], verbose = True)


# Show stuff to the screen if the user has entered a prompt
if prompt:
    response = sequential_chain({'topic': prompt})
    st.write(response['title'])
    st.write(response['script'])