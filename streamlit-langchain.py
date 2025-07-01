from youtube_transcript_api import YouTubeTranscriptApi
from pytube import YouTube
import wikipediaapi

from pathlib import Path
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
import streamlit as st
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

import os
from langchain import hub

from dotenv import load_dotenv
from langchain_openai.chat_models import ChatOpenAI
from langchain_community.document_loaders import WebBaseLoader
import bs4  # BeautifulSoup for parsing HTML

load_dotenv()  # take environment variables

# from .env file
# Load environment variables from .env file

token = os.getenv("SECRET")  # Replace with your actual token
endpoint = "https://models.github.ai/inference"
model = "openai/gpt-4.1-nano"

def read_file(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()

def transcribe_youtube_video(video_url):
    video_id = YouTube(video_url).video_id
    transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=['en', 'lt'])
    return "\n".join([t["text"] for t in transcript])

def fetch_wikipedia_content(wikipedia_url):
    try:
        parts = wikipedia_url.split('/')
        lang = parts[2].split('.')[0]      # e.g., 'lt'
        title = parts[-1]                  # e.g., 'Vilnius'

        # Add a proper user-agent
        email = os.getenv("WIKI_USER_EMAIL")  
        user_agent = f"MyStreamlitApp/1.0 ('{email}')"  
        wiki = wikipediaapi.Wikipedia(
            language=lang,
            user_agent=user_agent
        )

        page = wiki.page(title)
        if not page.exists():
            return f"⚠️ Wikipedia page '{title}' not found in language '{lang}'."
        return page.text
    
    except Exception as e:
        return f"Error extracting Wikipedia content: {str(e)}"

transcript_text = transcribe_youtube_video("https://www.youtube.com/watch?v=o0PmNtsqCzA")
text_from_file = read_file("Vilnius.txt")
text_from_wikipedia = fetch_wikipedia_content("https://lt.wikipedia.org/wiki/Vilnius")                                           

#loader = WebBaseLoader(
#    web_paths=("https://lt.wikipedia.org/wiki/Vilnius",),
#    bs_kwargs=dict(
#        parse_only=bs4.SoupStrainer(
#            class_=("mw-parser-output", "mw-heading")
#        )
#    ),
#)
#docs = loader.load()

st.title("Vilnius Guide RAG")
st.subheader("📖 Combined sources")

combined_text = f"{text_from_wikipedia}\n\n{transcript_text}\n\n{text_from_file}"
st.text_area("🎬 Combined text", combined_text, height=300)

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=10)
splits = text_splitter.create_documents([combined_text])

vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings(
    model="text-embedding-3-small",
    base_url="https://models.inference.ai.azure.com",
    api_key=token, # type: ignore
))

retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
prompt = hub.pull("rlm/rag-prompt")

def format_docs(docs):
    print(docs)
    return "\n\n".join(doc.page_content for doc in docs)

def generate_response(input_text):
    llm = ChatOpenAI(base_url=endpoint, temperature=0.7, api_key=token, model=model)

    fetched_docs = vectorstore.search(input_text, search_type="similarity", k=3)

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
        )
    
    
    st.info(rag_chain.invoke(input_text))
    
    st.subheader("📚 Sources")
    for i, doc in enumerate(fetched_docs, 1):
        with st.expander(f"Source {i}"):
            st.write(f"**Content:** {doc.page_content}")
            
with st.form("my_form"):
    text = st.text_area(
        "Enter text:",
        "What coordinates of Vilnius?",
    )
    submitted = st.form_submit_button("Submit")
    if submitted:
        generate_response(text)