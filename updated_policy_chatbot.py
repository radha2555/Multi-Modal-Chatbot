import os
import warnings
import logging
import gradio as gr
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import speech_recognition as sr
from pydub import AudioSegment
import tempfile
from io import StringIO
from dotenv import load_dotenv  # <--- Load env file

# Load environment variables
load_dotenv()

# Phase 2 - LLM for RAG
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

# Phase 3 - Vectorstore and Retrieval
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import TextLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.chains import RetrievalQA

# Disable warnings and info logs
warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)

# Function to convert speech to text
def speech_to_text(audio_file):
    recognizer = sr.Recognizer()
    with sr.AudioFile(audio_file) as source:
        audio_data = recognizer.record(source)
    try:
        text = recognizer.recognize_google(audio_data)
    except sr.UnknownValueError:
        text = "Sorry, I couldn't understand the audio."
    except sr.RequestError:
        text = "Speech recognition service unavailable."
    return text

# Function to extract text content from uploaded files
def extract_text_from_files(files):
    texts = []
    for file_path in files:
        with open(file_path, 'r', encoding='utf-8') as file:
            texts.append(file.read())
    return "\n".join(texts)

# Function to initialize vectorstore from files
def get_vectorstore(files):
    document_text = extract_text_from_files(files)
    with tempfile.NamedTemporaryFile(delete=False, mode='w', encoding='utf-8') as temp_file:
        temp_file.write(document_text)
        temp_file_path = temp_file.name
    loaders = [TextLoader(file_path=temp_file_path)]
    index = VectorstoreIndexCreator(
        embedding=HuggingFaceEmbeddings(model_name='all-MiniLM-L12-v2'),
        text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    ).from_loaders(loaders)
    os.remove(temp_file_path)
    return index.vectorstore

# Function to generate response using RAG
def get_answer(files, prompt):
    try:
        vectorstore = get_vectorstore(files)
        if vectorstore is None:
            return "Failed to load documents from the uploaded files."

        groq_sys_prompt = ChatPromptTemplate.from_template(
            """You are very smart and precise. Answer the question: {user_prompt}.
            Start the answer directly."""
        )

        model = "llama3-8b-8192"
        groq_chat = ChatGroq(
            groq_api_key=os.getenv("GROQ_API_KEY"),
            model_name=model
        )

        chain = RetrievalQA.from_chain_type(
            llm=groq_chat,
            chain_type='stuff',
            retriever=vectorstore.as_retriever(search_kwargs={'k': 3}),
            return_source_documents=True
        )

        result = chain({"query": prompt})
        return result["result"]
    except Exception as e:
        return f"Error: {str(e)}"

# Function to handle both text and voice queries using RAG
def chatbot_interface(files, prompt):
    return get_answer(files, prompt)

def chatbot_voice(audio_file, files):
    text_input = speech_to_text(audio_file)
    response = get_answer(files, text_input)
    return text_input, response

# Create Gradio UI
with gr.Blocks() as demo:
    gr.Markdown("## 📢 Multi-Modal Chatbot (Text, Voice, and RAG)")

    with gr.Row():
        file_input = gr.Files(label="Upload .txt files", type="filepath")
        text_input = gr.Textbox(label="Enter your message")
        text_button = gr.Button("Send")
    text_output = gr.Textbox(label="Chatbot Response")

    with gr.Row():
        audio_input = gr.Audio(type="filepath", label="Speak to the chatbot")
        voice_button = gr.Button("Send Voice")
    voice_transcript = gr.Textbox(label="Your Speech (Converted to Text)")
    voice_output = gr.Textbox(label="Chatbot Response")

    text_button.click(chatbot_interface, inputs=[file_input, text_input], outputs=text_output)
    voice_button.click(chatbot_voice, inputs=[audio_input, file_input], outputs=[voice_transcript, voice_output])

# Launch the Gradio UI without public share
demo.launch()
