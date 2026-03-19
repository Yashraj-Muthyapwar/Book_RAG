import streamlit as st
import os
import re
import base64
import shutil
from io import BytesIO
from PIL import Image
from dotenv import load_dotenv
import chromadb

# Docling Imports
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling_core.types.doc.document import ImageRefMode

# LlamaIndex Imports
from llama_index.core import Document, VectorStoreIndex, StorageContext, Settings, PromptTemplate
from llama_index.core.node_parser import MarkdownNodeParser
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding

# ==========================================
# 1. SETUP & CONFIGURATION
# ==========================================
load_dotenv()
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGES_DIR = os.path.join(BASE_DIR, "images")
DB_DIR = os.path.join(BASE_DIR, "chroma_db")

os.makedirs(IMAGES_DIR, exist_ok=True)
os.makedirs(DB_DIR, exist_ok=True)

# Configure Gemini Models
Settings.llm = GoogleGenAI(model="gemini-3.1-flash-lite-preview", api_key=os.environ.get("GEMINI_API_KEY"))
Settings.embed_model = GoogleGenAIEmbedding(model_name="gemini-embedding-2-preview", api_key=os.environ.get("GEMINI_API_KEY"))

st.set_page_config(page_title="Multimodal Book RAG", page_icon="📚", layout="wide")

# Text-only prompt (UI handles images separately)
QA_PROMPT_TMPL = (
    "You are an expert technical tutor. Context information is below.\n"
    "---------------------\n"
    "{context_str}\n"
    "---------------------\n"
    "Given the context information, answer the user's question clearly and accurately.\n"
    "CRITICAL: Do NOT include any markdown image links (e.g., ![alt](...)) in your final answer. "
    "Just provide the text explanation. The system will handle the visuals separately.\n"
    "Question: {query_str}\n"
    "Answer: "
)
qa_prompt = PromptTemplate(QA_PROMPT_TMPL)

# ==========================================
# 2. DATABASE CONNECTION
# ==========================================
@st.cache_resource
def get_database_connection():
    db = chromadb.PersistentClient(path=DB_DIR)
    chroma_collection = db.get_or_create_collection("databricks_lakehouse")
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    return db, chroma_collection, vector_store, storage_context

db_client, chroma_collection, vector_store, storage_context = get_database_connection()

if 'index' not in st.session_state:
    if chroma_collection.count() > 0:
        st.session_state['index'] = VectorStoreIndex.from_vector_store(
            vector_store=vector_store,
            storage_context=storage_context
        )
    else:
        st.session_state['index'] = None

# ==========================================
# 3. HELPER FUNCTIONS
# ==========================================
def clear_database():
    try:
        db_client.delete_collection("databricks_lakehouse")
    except Exception:
        pass
    if os.path.exists(IMAGES_DIR):
        shutil.rmtree(IMAGES_DIR)
        os.makedirs(IMAGES_DIR)
    st.session_state.clear()
    st.cache_resource.clear()
    st.rerun()

def get_responsive_image_html(image_path):
    with open(image_path, "rb") as f:
        data = base64.b64encode(f.read()).decode("utf-8")
        ext = image_path.split('.')[-1].lower()
        mime = f"image/{ext}" if ext in ['png', 'jpg', 'jpeg'] else 'image/png'
        filename = os.path.basename(image_path)
        
        html = f"""<div style="margin: 15px 0;">
<img src="data:{mime};base64,{data}" alt="Retrieved Diagram" style="max-width: 100%; width: auto; height: auto; border-radius: 8px; box-shadow: 0 4px 8px rgba(0,0,0,0.1);">
<br>
<a href="data:{mime};base64,{data}" download="{filename}" style="text-decoration: none; color: #ff4b4b; font-weight: bold; font-size: 0.9em;">
⬇️ Download Full-Resolution Diagram
</a>
</div>"""
        return html

# ==========================================
# 4. PROCESSING PIPELINE
# ==========================================
def process_pdf_pipeline(uploaded_file):
    temp_pdf_path = os.path.join(BASE_DIR, "temp_upload.pdf")
    
    with open(temp_pdf_path, "wb") as f:
        f.write(uploaded_file.getvalue())

    with st.status("Processing Document Pipeline...", expanded=True) as status:
        st.write("📄 **Step 1:** Parsing PDF and extracting high-res images with Docling...")
        
        pipeline_options = PdfPipelineOptions()
        pipeline_options.generate_picture_images = True
        pipeline_options.images_scale = 3.0 
        
        converter = DocumentConverter(
            allowed_formats=[InputFormat.PDF],
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
            }
        )
        
        result = converter.convert(temp_pdf_path)
        md_content = result.document.export_to_markdown(image_mode=ImageRefMode.EMBEDDED)

        st.write("🖼️ **Step 2:** Saving visuals to disk with decorative image filtering...")
        pattern = r"!\[(.*?)\]\(data:image/(.*?);base64,(.*?)\)"
        matches = list(re.finditer(pattern, md_content, re.DOTALL))
        
        # Tracking counts for logging
        total_found = len(matches)
        saved_count = 0

        existing_images_count = len([f for f in os.listdir(IMAGES_DIR) if os.path.isfile(os.path.join(IMAGES_DIR, f))])
        
        clean_content = md_content
        for i, match in enumerate(matches):
            alt_text = match.group(1)
            img_format = match.group(2)
            base64_data = match.group(3)
            
            img_bytes = base64.b64decode(base64_data)
            image = Image.open(BytesIO(img_bytes))
            
            # --- HEURISTIC FILTER: Skip decorative animal icons ---
            width, height = image.size
            if width < 300 or height < 300:
                # Remove the markdown link from text so it's not indexed without an image
                clean_content = clean_content.replace(match.group(0), "", 1)
                continue
            
            saved_count += 1
            image_filename = f"image_{existing_images_count + saved_count}.{img_format if img_format else 'png'}"
            image_filepath = os.path.join(IMAGES_DIR, image_filename)
            
            if image.mode in ("RGBA", "P") and img_format in ("jpeg", "jpg"):
                image = image.convert("RGB")
                
            image.save(image_filepath)
            
            relative_image_path = f"images/{image_filename}"
            replacement_string = f"![{alt_text}]({relative_image_path})"
            clean_content = clean_content.replace(match.group(0), replacement_string, 1)

        st.write(f"Done! Extracted {saved_count} technical diagrams (Filtered {total_found - saved_count} decorative icons).")

        st.write("🧩 **Step 3:** Chunking document hierarchically using LlamaIndex...")
        document = Document(text=clean_content, metadata={"source": uploaded_file.name})
        parser = MarkdownNodeParser()
        raw_nodes = parser.get_nodes_from_documents([document])

        nodes = [node for node in raw_nodes if node.text and node.text.strip()]
        st.write(f"Successfully generated {len(nodes)} hierarchical chunks (nodes).")

        st.write("🧠 **Step 4:** Embedding nodes and saving to ChromaDB...")
        if st.session_state.get('index') is None:
            st.session_state['index'] = VectorStoreIndex(nodes, storage_context=storage_context)
        else:
            st.session_state['index'].insert_nodes(nodes)
            
        status.update(label="✅ Pipeline Complete!", state="complete", expanded=False)
        
    if os.path.exists(temp_pdf_path):
        os.remove(temp_pdf_path)

# ==========================================
# 5. STREAMLIT UI & CHAT
# ==========================================
st.title("📚 Multimodal Agentic RAG")
st.markdown("Upload a technical PDF. The system automatically filters out decorative icons and stores high-res diagrams.")

with st.sidebar:
    st.header("Document Ingestion")
    uploaded_file = st.file_uploader("Upload a PDF chapter", type=["pdf"])
    
    if uploaded_file and st.button("Process Document"):
        process_pdf_pipeline(uploaded_file)
        
    st.divider()
    if st.session_state.get('index') is not None:
        try:
            st.success(f"Database Active: {chroma_collection.count()} vectors loaded.")
        except:
            pass
    else:
        st.warning("Database Empty. Please process a document.")
        
    st.divider()
    st.header("Maintenance")
    if st.button("🗑️ Clear Database & Reset", type="primary"):
        clear_database()

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        
        if "images" in message and message["images"]:
            for img in message["images"]:
                st.markdown(get_responsive_image_html(img), unsafe_allow_html=True)

        if "debug_nodes" in message and message["debug_nodes"]:
            with st.expander("🔍 Debug: View Retrieved Source Chunks"):
                for i, node_text in enumerate(message["debug_nodes"]):
                    st.markdown(f"**Chunk {i+1}**")
                    st.code(node_text)

if prompt := st.chat_input("Ask a question about your document..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    if st.session_state.get('index') is not None:
        engine = st.session_state['index'].as_query_engine(similarity_top_k=3)
        engine.update_prompts({"response_synthesizer:text_qa_template": qa_prompt})
        
        with st.chat_message("assistant"):
            with st.spinner("Searching the vector space and generating an answer..."):
                try:
                    response = engine.query(prompt)
                    answer_text = str(response)
                    
                    clean_answer = re.sub(r"!\[.*?\]\(.*?\)", "", answer_text).strip()
                    st.markdown(clean_answer)
                    
                    valid_images = []
                    seen_images = set()
                    debug_nodes_text = [] 
                    
                    for node in response.source_nodes:
                        debug_nodes_text.append(node.node.text)
                        
                        matches = re.findall(r"!\[.*?\]\((images/.*?)\)", node.node.text)
                        for img_path in matches:
                            if img_path not in seen_images:
                                seen_images.add(img_path)
                                full_path = os.path.join(BASE_DIR, img_path)
                                
                                if os.path.exists(full_path):
                                    valid_images.append(full_path)
                                    st.markdown(get_responsive_image_html(full_path), unsafe_allow_html=True)
                    
                    with st.expander("🔍 Debug: View Retrieved Source Chunks"):
                        for i, node_text in enumerate(debug_nodes_text):
                            st.markdown(f"**Chunk {i+1}**")
                            st.code(node_text)
                    
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": clean_answer,
                        "images": valid_images, 
                        "debug_nodes": debug_nodes_text
                    })
                except Exception as e:
                    st.error(f"An error occurred during the query: {str(e)}")
    else:
        st.error("Please upload and process a document first!")
