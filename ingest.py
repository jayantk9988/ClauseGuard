from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma


DATA_DIR = "data"
DB_DIR = "chroma_db"


loader = PyPDFDirectoryLoader(DATA_DIR)
documents = loader.load()

if not documents:
    raise ValueError("No PDF documents found in the data folder.")


text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

chunks = text_splitter.split_documents(documents)


embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory=DB_DIR
)

print(f"Ingestion complete. Created {len(chunks)} chunks.")