import streamlit as st
from streamlit_chat import message
from dotenv import load_dotenv
import os
from langchain.chat_models import ChatOpenAI
from langchain.schema import (
    SystemMessage,
    HumanMessage,
    AIMessage,
)
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.callbacks import get_openai_callback

def init():
    load_dotenv()
    if os.getenv("OPENAI_API_KEY") is None or os.getenv("OPENAI_API_KEY") == "":
        print("OPENAI_API_KEY is not set")
        exit(1)
    else:
        print("OPENAI_API_KEY is set")

    st.set_page_config(
        page_title="My AI Assistant",
        page_icon=":material/chat:",
    )

def main():
    init()

    chat = ChatOpenAI(temperature=0)

    if "messages" not in st.session_state:
        st.session_state.messages = [
            SystemMessage(content="You are a helpful assistant.")
        ]
    if "pdf_mode" not in st.session_state:
        st.session_state.pdf_mode = False
    if "knowledge_base" not in st.session_state:
        st.session_state.knowledge_base = None

    col1, col2 = st.columns([1, 6])
    with col1:
        st.image("https://cdn-icons-png.freepik.com/512/3593/3593684.png", width=60)
    with col2:
        st.header("My AI Assistant")

    with st.sidebar:
        mode = st.radio("Select Mode", ("Chat", "PDF Chat"))
        if mode == "PDF Chat":
            st.session_state.pdf_mode = True
            pdf = st.file_uploader("Upload your PDF", type="pdf")
            if pdf is not None:
                pdf_reader = PdfReader(pdf)
                text = ""
                for page in pdf_reader.pages:
                    text += page.extract_text()
                text_splitter = CharacterTextSplitter(
                    separator="\n",
                    chunk_size=1000,
                    chunk_overlap=200,
                    length_function=len,
                )
                chunks = text_splitter.split_text(text)
                embeddings = OpenAIEmbeddings()
                st.session_state.knowledge_base = FAISS.from_texts(chunks, embeddings)
        else:
            st.session_state.pdf_mode = False
            st.session_state.knowledge_base = None

        user_input = st.text_input("Enter your Query", key="user_input")

        if user_input:
            if st.session_state.pdf_mode and st.session_state.knowledge_base:
                docs = st.session_state.knowledge_base.similarity_search(user_input)
                llm = OpenAI()
                chain = load_qa_chain(llm, chain_type="stuff")
                with get_openai_callback() as callback:
                    response = chain.run(input_documents=docs, question=user_input)
                    print(callback)
                st.session_state.messages.append(HumanMessage(content=user_input))
                st.session_state.messages.append(AIMessage(content=response)) # Corrected Line.
            else:
                st.session_state.messages.append(HumanMessage(content=user_input))
                with st.spinner("Thinking..."):
                    response = chat(st.session_state.messages)
                st.session_state.messages.append(AIMessage(content=response.content))

    messages = st.session_state.get("messages", [])
    for i, msg in enumerate(messages[1:]):
        if i % 2 == 0:
            message(msg.content, is_user=True, key=str(i) + "_user")
        else:
            message(msg.content, is_user=False, key=str(i) + "_ai")

if __name__ == "__main__":
    main()