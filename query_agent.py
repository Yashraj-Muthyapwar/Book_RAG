import os
from dotenv import load_dotenv
import chromadb
from llama_index.core import VectorStoreIndex, Settings
from llama_index.vector_stores.chroma import ChromaVectorStore

# NEW: Import the Google GenAI wrappers
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding

# 1. Load the API Key
load_dotenv()

# 2. Configure LlamaIndex with the exact same models used for ingestion
Settings.llm = GoogleGenAI(
    model="gemini-3.1-flash-lite-preview", 
    api_key=os.environ.get("GEMINI_API_KEY")
)

Settings.embed_model = GoogleGenAIEmbedding(
    model_name="gemini-embedding-2-preview", 
    api_key=os.environ.get("GEMINI_API_KEY")
)

# 3. Set up paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_DIR = os.path.join(BASE_DIR, "chroma_db")

# 4. Connect to your existing ChromaDB and the specific collection
print("Connecting to the local ChromaDB database...")
try:
    db = chromadb.PersistentClient(path=DB_DIR)
    # We use get_collection here because we expect it to already exist
    chroma_collection = db.get_collection("databricks_lakehouse")
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
except Exception as e:
    print(f"Error connecting to the database. Did you run build_multimodal_index.py first? Error: {e}")
    exit()

# 5. Load the index directly from the vector store
print("Loading vector index...")
index = VectorStoreIndex.from_vector_store(
    vector_store=vector_store,
)

# 6. Initialize the Query Engine
# similarity_top_k=3 tells the retriever to pull the 3 most relevant chunks
query_engine = index.as_query_engine(similarity_top_k=3)

print("\n✅ Multimodal RAG Agent is ready! Type 'exit' to quit.\n")

# 7. Start an interactive terminal loop
while True:
    user_query = input("Ask a question about the Delta Lake chapter: ")
    
    if user_query.lower() in ['exit', 'quit']:
        print("Shutting down agent...")
        break
        
    if not user_query.strip():
        continue
        
    print("\nSearching the vector space and generating an answer...")
    try:
        response = query_engine.query(user_query)
        print(f"\n🤖 Answer:\n{response}\n")
            
    except Exception as e:
        print(f"\n❌ An error occurred during the query: {e}\n")
        
    print("-" * 60 + "\n")
