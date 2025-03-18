import streamlit as st
import os
from openai import OpenAI
from dotenv import load_dotenv
from streamlit_chat import message
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

def generate_images(img_description):
    """Generates an image using DALL-E 3 based on the provided description."""
    load_dotenv()
    openai_api_key = os.getenv("OPENAI_API_KEY")

    if not openai_api_key:
        st.error("OpenAI API key not found. Please set the OPENAI_API_KEY environment variable.")
        return None

    client = OpenAI(api_key=openai_api_key)

    try:
        image_response = client.images.generate(
            model="dall-e-3",
            prompt="Generate an image of a " + img_description,
            size="1024x1024",
            quality="standard",
            n=1,
        )
        image_url = image_response.data[0].url
        return image_url
    except Exception as e:
        st.error(f"An error occurred: {e}")
        return None

def chat_pdf_app():
    load_dotenv()
    if os.getenv("OPENAI_API_KEY") is None or os.getenv("OPENAI_API_KEY") == "":
        st.error("OPENAI_API_KEY is not set")
        return

    st.title("📚 ChatPDF Assistant")
    st.markdown("**Ask questions about your uploaded PDF documents.**")

    if "messages" not in st.session_state:
        st.session_state.messages = [
            SystemMessage(content="You are a helpful assistant that answers question based on uploaded PDF.")
        ]
    if "knowledge_base" not in st.session_state:
        st.session_state.knowledge_base = None

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

    user_input = st.text_input("Enter your Query", key="user_input")

    if user_input and st.session_state.knowledge_base:
        docs = st.session_state.knowledge_base.similarity_search(user_input)
        llm = OpenAI()
        chain = load_qa_chain(llm, chain_type="stuff")
        with get_openai_callback() as callback:
            response = chain.run(input_documents=docs, question=user_input)
            print(callback)
        st.session_state.messages.append(HumanMessage(content=user_input))
        st.session_state.messages.append(AIMessage(content=response))

    messages = st.session_state.get("messages", [])
    for i, msg in enumerate(messages[1:]):
        if i % 2 == 0:
            message(msg.content, is_user=True, key=str(i) + "_user")
        else:
            message(msg.content, is_user=False, key=str(i) + "_ai")

def own_chatgpt_app():
    load_dotenv()
    if os.getenv("OPENAI_API_KEY") is None or os.getenv("OPENAI_API_KEY") == "":
        st.error("OPENAI_API_KEY is not set")
        return

    st.title("💬 ChatGPT Assistant")
    st.markdown("**Have a conversation with an AI assistant.**")

    chat = ChatOpenAI(temperature=0)

    if "messages" not in st.session_state:
        st.session_state.messages = [
            SystemMessage(content="You are a helpful assistant.")
        ]

    user_input = st.text_input("Enter your Query", key="user_input")

    if user_input:
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

def main():
    st.set_page_config(
        page_title="AI Tools",
        page_icon=":magic_wand:",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.sidebar.title("AI Tools")
    app_mode = st.sidebar.radio("Choose an app", ["Chatter", "PaperAI","Imagery"])

    if app_mode == "Imagery":
        st.title("🎨 Image Generator Assistant")
        st.markdown("**Create stunning images from your text descriptions!**")
        img_description = st.text_input("Enter your image description:", placeholder="A futuristic cityscape at sunset")
        if st.button("Generate Image"):
            if img_description:
                with st.spinner("Generating image..."):
                    image_url = generate_images(img_description)
                if image_url:
                    st.image(image_url, caption=img_description)
                    st.success("Image generated successfully!")
                else:
                    st.error("Image generation failed. Please check your API key and description.")
            else:
                st.warning("Please enter an image description.")

    elif app_mode == "PaperAI":
        chat_pdf_app()

    elif app_mode == "Chatter":
        own_chatgpt_app()

if __name__ == "__main__":
    main()