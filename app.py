#pip install youtube-transcript-api,chromadb,langchain,streamlit
from langchain.document_loaders import YoutubeLoader
from langchain import OpenAI
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.vectorstores import Chroma
from allapis import apikey
import streamlit as st
import os
import pyttsx3

# Initialize the text-to-speech engine
engine = pyttsx3.init()
# Set the voice to female voice
voice_id = 'com.apple.speech.synthesis.voice.karen'
engine.setProperty('voice', voice_id)

# Display the YouTube logo
#st.image("/Users/shantanushirishramteke/Desktop/Visual Studio Journey/AskTube/youtube-round-2.1024x1024.png", width=100)
st.image("images/youtube-round-2.1024x1024.png", width=100)

# Display the Page Title
st.title('AskTube')

# Set up OpenAI API credentials
os.environ["OPENAI_API_KEY"] = apikey

llm = OpenAI(temperature=0)

prompt = st.text_input("Enter Youtube URL")

if prompt:
    loader = YoutubeLoader.from_youtube_url(prompt, add_video_info=False)
    docs = loader.load()

    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)

    split_docs = text_splitter.split_documents(docs)

    embeddings = OpenAIEmbeddings(openai_api_key= apikey)
    doc_search = Chroma.from_documents(docs,embeddings)
    chain = RetrievalQA.from_chain_type(llm=llm,chain_type="stuff",retriever=doc_search.as_retriever())

    query = st.text_input("So what do you want to know:")

    if st.button("Tell me"):
        answer = chain.run(query)
        st.write("Answer:")
        st.write(answer)
        # Convert the answer to speech and play it
        engine.say(answer)
        engine.runAndWait()