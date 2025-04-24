
## 📢 Multi-Modal Chatbot with Text, Voice, and RAG

This project is a versatile chatbot that accepts both **text** and **voice** input, combining **voice recognition** with **retrieval-augmented generation (RAG)** for smarter, context-driven responses. It uses **Groq's LLM**, **LangChain**, and **Gradio** to provide an interactive user experience, while also enabling **vector-based search** over documents you upload.

---

### ✨ Key Features
- 🗣️ **Voice Recognition** – Convert speech into text and query the chatbot.
- 📚 **Document-based Context** – Upload `.txt` files to provide a knowledge base for smarter responses.
- 💡 **Contextual Answers** – Retrieves relevant information from uploaded documents via **LangChain's RAG** integration.
- 🖥️ **User-Friendly Interface** – Gradio-based UI that supports both text and voice input.

---

### 🛠️ Installation & Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/your-username/your-repo-name.git
   cd your-repo-name
   ```

2. **Create a virtual environment (optional but recommended)**:
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   Create a `requirements.txt` file and add the following:

   ```txt
   gradio
   torch
   transformers
   pydub
   SpeechRecognition
   python-dotenv
   langchain
   langchain-groq
   ```

   Then run:

   ```bash
   pip install -r requirements.txt
   ```

4. **Set up your `.env` file**:
   In the root directory, create a `.env` file with your **Groq API Key**:

   ```bash
   touch .env
   ```

   Inside `.env`:

   ```env
   GROQ_API_KEY=your_groq_api_key_here
   ```

---

### 🏃‍♂️ Running the App

Once the environment is set up, just run:

```bash
python your_script_name.py
```

This will start a **Gradio interface** locally. The chatbot will be accessible in your browser for both **text and voice** input.

---

### 🧠 How It Works

1. **Voice Input**: 
   - Speak into the microphone.
   - The system converts the audio to text and queries the chatbot with it.
   - The chatbot uses **LangChain's retrieval mechanism** to search through your uploaded documents and returns an informed answer.

2. **Text Input**: 
   - Simply type a message.
   - You can upload one or more `.txt` files, which will be used as the knowledge base for context-based responses.

3. **RAG**: 
   - The system uses **Retrieval-Augmented Generation (RAG)** to combine the power of large language models with document search, ensuring answers are relevant and informed by the uploaded content.

---

### 🧑‍💻 Tech Stack

| **Component**          | **Library**                 |
|------------------------|-----------------------------|
| **LLM**                | `langchain-groq`            |
| **Voice Recognition**  | `speech_recognition`        |
| **Embeddings**         | `HuggingFaceEmbeddings`     |
| **Document Indexing**  | `langchain` (Vectorstore)   |
| **UI**                 | `gradio`                    |
| **Environment**        | `python-dotenv`             |

---

### 📂 File Uploads

- **Supported files**: Only **.txt** files are accepted for document uploads.
- The chatbot will retrieve information from these files to improve its answers.

---

### 🚨 Important Notes

- This application is **local only** — no public URL is generated. (You can change this if you prefer by setting `share=True` in the code.)
- Ensure the audio files are in a compatible format (WAV is recommended).
- **GROQ API Key** is required to use the chatbot — make sure you add it to the `.env` file.

---

### 🤖 Acknowledgements

- [LangChain](https://www.langchain.com/)
- [Groq](https://groq.com/)
- [Gradio](https://gradio.app/)
- [HuggingFace](https://huggingface.co/)

---
