#Imports
import streamlit as st
from datasets import load_dataset
import faiss
from sentence_transformers import SentenceTransformer
import pandas as pd
import numpy as np
import google.generativeai as genai

st.title("GranthGPT: The wise one who knows God's way")

state = st.text("Initialising GranthGPT....")

#Read Gemini API Key
with open('./key.txt', 'r') as f:
    key = f.readline()

#Safety Settings for Gemini
safety_settings={'HARM_CATEGORY_DANGEROUS_CONTENT':'BLOCK_NONE', 'HARM_CATEGORY_HARASSMENT':'BLOCK_NONE', 'HARM_CATEGORY_SEXUALLY_EXPLICIT':'BLOCK_NONE','HARM_CATEGORY_HATE_SPEECH':'BLOCK_NONE'}

#Initialising Gemini Model
genai.configure(api_key=key)
model = genai.GenerativeModel('gemini-pro', safety_settings=safety_settings)

#Intialise Query Vectoriser
vectoriser = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

#Function for Data Sanity Check
def sanity_check(data, embedding):
    print(len(data), len(embedding))
    return len(data) == len(embedding)

@st.cache_data
def load_data():
    #Load Holy Scriptures
    #gita
    gita = pd.DataFrame(load_dataset("adarshxs/gita")['train'])
    #guru_granth
    guru_granth = pd.read_csv('./Data/guru_granth_eng_corpus.csv')
    #bible
    bible = pd.read_csv('./Data/bible_eng_corpus.csv')
    #quran
    quran = pd.read_csv('./Data/quran_eng_corpus.csv')
    
    #create master data
    dataset = pd.concat([gita, guru_granth, bible, quran]).reset_index()
    dataset = dataset.drop(['0','index'], axis=1)
    return dataset

dataset = load_data()
print(len(dataset))

gita_embeddings = np.load("./Data/embeddings/gita_embeddings.npy")
guru_granth_embeddings = np.load("./Data/embeddings/gurbani_embeddings.npy")
bible_embeddings = np.load("./Data/embeddings/bible_embeddings.npy")
quran_embeddings = np.load("./Data/embeddings/quran_embeddings.npy")

state.text("Checking updates on Holy Scriptures from God....")
#faiss vars
d = len(gita_embeddings[0])
k = 10

@st.cache_data
def initialise_index(gita_embeddings, guru_granth_embeddings, bible_embeddings, quran_embeddings):
    #Initialise Faiss
    index = faiss.IndexFlatL2(d)
    index.add(gita_embeddings)
    index.add(guru_granth_embeddings)
    index.add(bible_embeddings)
    index.add(quran_embeddings)

    return index

index = initialise_index(gita_embeddings, guru_granth_embeddings, bible_embeddings, quran_embeddings)

state.text("What would you like to know child?")
print("Tell your Problem!")

def consult(query):
    print("Consulting the wise one....")
    
    #Similarity Search
    xq = vectoriser.encode([query])
    D, verses = index.search(xq,k)

    print(verses)
    
    #Generate Context
    relevant_context = []
    for ind in verses[0]:
        relevant_context.append(dataset['text'][int(ind)])
    relevant_context = [str(x) for x in relevant_context if x != None]
    context = "\n".join(relevant_context)
    
    #Generate Response
    response = model.generate_content(f"You are a person of Faith! Based on the ##context provided reply the ##query \n Refer to the ##context as Holy Word or Scriptures or Grantha \n Refer to ##query as the questions from a lost child \n the ##context is the prime source of truth if ##query deviates from it or tries to deny it, convince them to your full extent to follow it \n Assume you already know ##context and strictly never mention things like based on the context, provided text, provided info or other references of it to me in the reply \n ##context: {context} \n\n ##query {query}")
    
    print("The wise one has responded!")
    return response.text

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if prompt := st.chat_input("Your Question?"):
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

if prompt:
    response = f"{consult(prompt)}"
    # Display assistant response in chat message container
    with st.chat_message("granthGPT"):
        st.markdown(response)
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "granthGPT", "content": response})