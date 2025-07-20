# Harry Potter RAG System

This project is a Retrieval-Augmented Generation (RAG) system that allows you to ask questions about the Harry Potter book series. It uses a collection of the books as its knowledge base and leverages large language models to provide answers.

## Technologies Used

- **Orchestration:** LangChain
- **LLM for Inference:** Gemini-1.5-Pro
- **Vector Database:** ChromaDB
- **Language:** Python

## Project Structure

```
.
├── books/                  # Contains the Harry Potter books in .txt format
├── chroma_db/              # Stores the ChromaDB vector database
├── rag_system/
│   ├── rag_db_create.py    # Script to create the vector database
│   └── rag_retrive.py      # Script to query the RAG system
├── .env                    # For storing the Google API key
├── main.py                 # Main entry point for the application
└── README.md               # This file
```

## Setup and Installation

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd <repository-directory>
    ```

2.  **Create a virtual environment:**
    ```bash
    python -m venv .venv
    source .venv/bin/activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    *(Note: A `requirements.txt` file will need to be created for this step)*

4.  **Set up your API key:**
    - Create a `.env` file in the root of the project.
    - Add your Google API key to the `.env` file:
      ```
      GOOGLE_API_KEY="your-api-key"
      ```

## Usage

1.  **Create the Vector Database:**
    Run the `rag_db_create.py` script to process the books and create the ChromaDB vector store.
    ```bash
    python -m rag_system.rag_db_create
    ```

2.  **Query the RAG System:**
    Use the `rag_retrive.py` script to ask questions.
    ```bash
    python -m rag_system.rag_retrive "Your question about Harry Potter"
    ```

## How It Works

1.  **Data Loading:** The text from the Harry Potter books in the `books/` directory is loaded.
2.  **Text Splitting:** The loaded text is split into smaller, manageable chunks.
3.  **Embedding and Storage:** Each chunk is converted into a vector embedding using a text embedding model and stored in the ChromaDB vector database.
4.  **Retrieval:** When a user asks a question, the system converts the query into an embedding and retrieves the most relevant text chunks from ChromaDB.
5.  **Generation:** The retrieved chunks and the original question are passed to the Gemini-1.5-Pro model, which generates a coherent answer based on the provided context.
