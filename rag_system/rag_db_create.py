import asyncio

def create_vec_db():
    # Fix: Ensure event loop is set (for gRPC async)
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        asyncio.set_event_loop(asyncio.new_event_loop())

    from langchain_google_genai import GoogleGenerativeAIEmbeddings
    from langchain_chroma import Chroma
    from langchain.text_splitter import CharacterTextSplitter
    from langchain_community.document_loaders import TextLoader
    import os
    from dotenv import load_dotenv
    load_dotenv()

    current_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(current_dir, "books")
    persistent_dir = os.path.join(current_dir, "chroma_db")

    embedder = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    if not os.path.exists(persistent_dir):
        if not os.path.exists(file_path):
            raise FileNotFoundError("No dataset is found.")

        books = [doc for doc in os.listdir(file_path) if doc.endswith(".txt")]
        documents = []

        for book in books:
            book_path = os.path.join(file_path, book)
            loader = TextLoader(book_path)
            book_docs = loader.load()
            for doc in book_docs:
                doc.metadata = {"source": book}
                documents.append(doc)

        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
        docs = text_splitter.split_documents(documents)

        print("No. of chunks:", len(docs))

        db = Chroma.from_documents(docs, embedder, persist_directory=persistent_dir)
        return "Created vector database on Harry Potter books."
    else:
        db = Chroma(persist_directory=persistent_dir, embedding_function=embedder)
        return "Vector database already exists and is loaded."
