# DocuChat-Your-AI-Companion

This Streamlit application combines the power of a conversational AI with the ability to query information from uploaded PDF documents.

## Features

* **Dual Mode:** Choose between a general AI chat mode and a PDF document query mode.
* **PDF Upload:** Upload PDF files to extract text and create a searchable knowledge base.
* **AI Chat:** Engage in general conversations with an AI assistant powered by OpenAI's language models.
* **PDF Querying:** Ask questions about the content of your uploaded PDFs and receive relevant answers.
* **Streamlit Interface:** User-friendly web interface built with Streamlit.
* **LangChain Integration:** Utilizes LangChain for document processing and question answering.
* **FAISS Vector Store:** Efficiently stores and retrieves document embeddings using FAISS.

## Prerequisites

* Python 3.8+
* Streamlit (`streamlit`)
* streamlit-chat (`streamlit-chat`)
* python-dotenv (`python-dotenv`)
* LangChain (`langchain`)
* OpenAI API Key (`openai`)
* PyPDF2 (`pypdf2`)

## Installation

1.  **Clone the repository:**

    ```bash
    git clone <repository_url>
    cd <repository_directory>
    ```

2.  **Create a virtual environment (recommended):**

    ```bash
    python -m venv venv
    source venv/bin/activate  # On macOS and Linux
    venv\Scripts\activate  # On Windows
    ```

3.  **Install dependencies:**

    ```bash
    pip install streamlit streamlit-chat python-dotenv langchain openai pypdf2 faiss-cpu
    ```

4.  **Set up your OpenAI API key:**

    * Create a `.env` file in the root directory of your project.
    * Add your OpenAI API key to the `.env` file:

        ```
        OPENAI_API_KEY=your_openai_api_key
        ```

    * **Important:** Do not commit your `.env` file to version control.

## Usage

1.  **Run the Streamlit application:**

    ```bash
    streamlit run app.py
    ```

    (Replace `app.py` with the name of your Python file if it's different.)

2.  **Open the application in your browser:**

    Streamlit will provide a URL (usually `http://localhost:8501`) that you can open in your web browser.

3.  **Start using the application:**

    * Select the desired mode ("Chat" or "PDF Chat") from the sidebar.
    * If in "PDF Chat" mode, upload a PDF file.
    * Enter your question or message in the text input field.
    * The AI's response will appear in the chat history.

## Code Explanation

* **`init()`:** Initializes the application by loading the OpenAI API key and setting up the Streamlit page configuration.
* **`main()`:**
    * Creates a `ChatOpenAI` instance for general chat.
    * Handles mode selection and PDF processing.
    * Creates a FAISS knowledge base for PDF querying.
    * Handles user input and generates responses using either the general chat or PDF querying logic.
    * Displays the chat history using `streamlit_chat.message`.

## Future Improvements

* **Error Handling:** Implement more robust error handling for API errors and file processing.
* **Context Management:** Improve context management for long conversations and large PDF documents.
* **UI Enhancements:** Enhance the user interface with features like progress indicators and better message formatting.
* **Multiple PDF Uploads:** Allow users to upload and query multiple PDF files.
* **More advanced System Messages:** Allow the user to change the system message.

## Contributing

Feel free to contribute to this project by submitting pull requests or opening issues.
