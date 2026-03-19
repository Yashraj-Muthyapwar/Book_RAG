import os
from dotenv import load_dotenv
import chromadb
from llama_index.core import VectorStoreIndex, StorageContext, Settings
from llama_index.vector_stores.chroma import ChromaVectorStore

# NEW: Import the Google GenAI wrappers
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding

# Import the generated nodes and base directory from your extraction script
from text_converter import nodes, BASE_DIR

# 1. Load the GEMINI_API_KEY from the .env file
load_dotenv()

# 2. Configure LlamaIndex to use the NEW Google GenAI SDK
Settings.llm = GoogleGenAI(
    model="gemini-2.0-flash", 
    api_key=os.environ.get("GEMINI_API_KEY")
)

Settings.embed_model = GoogleGenAIEmbedding(
    model_name="gemini-embedding-2-preview", 
    api_key=os.environ.get("GEMINI_API_KEY")
)

# 3. Set up the Persistent ChromaDB Client
DB_DIR = os.path.join(BASE_DIR, "chroma_db")
db = chromadb.PersistentClient(path=DB_DIR)

# Create a collection specifically for this book/chapter
chroma_collection = db.get_or_create_collection("databricks_lakehouse")

# 4. Bind ChromaDB to LlamaIndex
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
storage_context = StorageContext.from_defaults(vector_store=vector_store)

# 5. Ingest the Nodes into the Vector Store
print("Embedding nodes using the new gemini-embedding-2-preview model...")
index = VectorStoreIndex(
    nodes, 
    storage_context=storage_context,
    show_progress=True
)
print("Ingestion complete! Multimodal vectors saved to ChromaDB.")
