import os
from apikey import apikey

import streamlit as st
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SequentialChain

os.environ['OPENAI_API_KEY'] = apikey

st.title('Problem')
prompt_problem = st.text_input('What are you trying to solve?:')

st.title('Challenge #1')
prompt_challenge1 = st.text_input('What is one thing that makes solving this difficult:')

st.title('Challenge #2')
prompt_challenge2 = st.text_input('What is a second thing that makes solving this difficult:')

st.title('Challenge #3')
prompt_challenge3 = st.text_input('What is a third thing that makes solving this difficult:')

#Prompt templates
# title_template = PromptTemplate(
#     input_variables = ['topic'],
#     template = 'write me a youtube video title about {topic}'
# )

#Prompt templates
# script_template = PromptTemplate(
#     input_variables = ['title'],
#     template = 'write me a youtube video script based on this title Title: {title}'
# )

#Prompt templates
statement_template = PromptTemplate(
    input_variables = ['problem', 'challenge1', 'challenge2', 'challenge3'],
    template = 'Write me a problem statement based on this problem: {problem} and these challenges to solving the problem: {challenge1}, {challenge2}, {challenge3}'
)

#Prompt templates
title_template = PromptTemplate(
    input_variables = ['statement'],
    template = 'write me a project title about {statement}'
)

#Llms
llm = OpenAI(temperature=0.9)
statement_chain = LLMChain(llm=llm, prompt=statement_template, verbose=True, output_key='statement')
title_chain = LLMChain(llm=llm, prompt=title_template, verbose=True, output_key='title')
sequential_chain = SequentialChain(chains=[statement_chain, title_chain], input_variables=['problem', 'challenge1', 'challenge2', 'challenge3'], output_variables=['statement', 'title'], verbose = True)


# Show stuff to the screen if the user has entered a prompt
if prompt_problem:
    response = sequential_chain({'problem': prompt_problem, 'challenge1': prompt_challenge1, 'challenge2': prompt_challenge2, 'challenge3': prompt_challenge3})
    st.write(response['statement'])
    st.write(response['title'])