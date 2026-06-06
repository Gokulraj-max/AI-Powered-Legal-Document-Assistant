import streamlit as st
import os
import tempfile
import pandas as pd
from datetime import datetime
from utils.document_parser import parse_document
from utils.vector_store import VectorStore
from utils.llm_client import get_embedding, get_embeddings_batch, generate_answer

# Define layout and page title
st.set_page_config(
    page_title="LexiRAG | AI-Powered Legal Assistant",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom premium CSS styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&family=Playfair+Display:ital,wght@0,600;1,400&display=swap');
    
    /* Global Styles */
    .stApp {
        font-family: 'Outfit', sans-serif;
    }
    
    h1, h2, h3 {
        font-family: 'Playfair Display', serif;
        font-weight: 700;
    }
    
    /* Header style */
    .header-container {
        padding: 20px 0px;
        margin-bottom: 20px;
        border-bottom: 1px solid rgba(255, 255, 255, 0.1);
        display: flex;
        align-items: center;
        gap: 15px;
    }
    .main-title {
        font-size: 2.5rem;
        background: linear-gradient(135deg, #FFD700 0%, #D4AF37 50%, #B8860B 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0;
    }
    .subtitle {
        font-size: 1.1rem;
        color: #8A9A86;
        margin-top: -5px;
        font-style: italic;
    }

    /* Glassmorphic card styling for sources */
    .source-card {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-left: 4px solid #D4AF37;
        border-radius: 8px;
        padding: 12px 18px;
        margin: 8px 0;
        transition: all 0.2s ease-in-out;
    }
    .source-card:hover {
        transform: translateY(-2px);
        background: rgba(255, 255, 255, 0.05);
        border-color: rgba(212, 175, 55, 0.4);
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
    }
    .source-meta {
        font-size: 0.85rem;
        color: #B8860B;
        font-weight: 600;
        display: flex;
        justify-content: space-between;
        margin-bottom: 6px;
    }
    .source-text {
        font-size: 0.9rem;
        line-height: 1.5;
        font-style: italic;
        color: rgba(255, 255, 255, 0.85);
    }

    /* Stat Cards */
    .stat-card {
        background: rgba(255, 255, 255, 0.02);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 10px;
        padding: 20px;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        transition: transform 0.2s;
    }
    .stat-card:hover {
        transform: scale(1.02);
        border-color: rgba(255, 255, 255, 0.1);
    }
    .stat-val {
        font-size: 2.2rem;
        font-weight: 700;
        color: #D4AF37;
        margin-bottom: 5px;
    }
    .stat-lbl {
        font-size: 0.9rem;
        color: #8A9A86;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
</style>
""", unsafe_allow_html=True)

# Initialize vector store
db_file_path = "C:\\Projects\\AI-Powered-Legal-Document-Assistant\\legal_assistant.db"
vector_store = VectorStore(db_path=db_file_path)

# Page Header
st.markdown("""
<div class="header-container">
    <div>
        <h1 class="main-title">⚖️ LexiRAG</h1>
        <div class="subtitle">AI-Powered Legal Document Assistant</div>
    </div>
</div>
""", unsafe_allow_html=True)

# Initialize session states
if "messages" not in st.session_state:
    st.session_state.messages = []

# Sidebar Configurations
with st.sidebar:
    st.markdown("### ⚙️ API Configuration")
    
    # Check for Gemini API key in environment
    env_api_key = os.getenv("GEMINI_API_KEY", "")
    api_key_input = st.text_input(
        "Gemini API Key",
        value=env_api_key,
        type="password",
        help="Provide your Google Gemini API Key. Get one from Google AI Studio."
    )
    
    st.markdown("---")
    st.markdown("### 🛠️ Model & RAG Settings")
    
    selected_model = st.selectbox(
        "LLM Model",
        options=["gemini-2.5-flash", "gemini-2.5-pro"],
        index=0,
        help="Gemini 2.5 Flash is recommended for speed. Pro is recommended for complex reasoning."
    )
    
    chunk_size = st.slider("Chunk Size (characters)", min_value=300, max_value=2000, value=1000, step=100)
    chunk_overlap = st.slider("Chunk Overlap (characters)", min_value=50, max_value=500, value=200, step=50)
    
    st.markdown("---")
    st.markdown("### 🔍 Retrieval Configuration")
    
    top_k = st.slider("Retrieve Top-K Chunks", min_value=2, max_value=10, value=5)
    min_score = st.slider("Minimum Similarity Match", min_value=0.1, max_value=0.9, value=0.3, step=0.05)
    temperature = st.slider("Temperature", min_value=0.0, max_value=1.0, value=0.2, step=0.1)

    # Get DB Stats for sidebar display
    db_stats = vector_store.get_stats()
    
    st.markdown("---")
    st.markdown("### 📊 Database Overview")
    st.markdown(f"**Total Documents:** `{db_stats['total_documents']}`")
    st.markdown(f"**Total Vector Chunks:** `{db_stats['total_chunks']}`")
    
    # System Controls
    st.markdown("### ⚠️ Danger Zone")
    if st.button("Reset Vector Database", type="secondary", use_container_width=True):
        vector_store.reset_database()
        st.session_state.messages = []
        st.success("Vector database reset successfully! Chat history cleared.")
        st.rerun()

# Define Tabs
tab_chat, tab_library, tab_analytics = st.tabs([
    "💬 Chat Assistant", 
    "📂 Document Library", 
    "📈 Visual Analytics"
])

# ----------------- TAB 1: CHAT ASSISTANT -----------------
with tab_chat:
    # Check if database is empty
    if db_stats["total_documents"] == 0:
        st.info("👋 Welcome to LexiRAG! Get started by uploading legal documents (PDFs, DOCXs, TXTs) in the **Document Library** tab.")
    else:
        # Display chat history
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                
                # If assistant message contains sources, display them in an expander
                if message["role"] == "assistant" and message.get("sources"):
                    with st.expander("🔍 Citations & Retrieved Sources"):
                        for idx, src in enumerate(message["sources"]):
                            st.markdown(f"""
                            <div class="source-card">
                                <div class="source-meta">
                                    <span>📄 {src['filename']} ({src['location']})</span>
                                    <span>Score: {src['score']:.2f}</span>
                                </div>
                                <div class="source-text">"{src['text']}"</div>
                            </div>
                            """, unsafe_allow_html=True)

        # Chat Input
        if user_query := st.chat_input("Ask a question about your uploaded legal documents..."):
            # Display user message
            st.chat_message("user").markdown(user_query)
            st.session_state.messages.append({"role": "user", "content": user_query})
            
            # Show spinner and process RAG
            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                with st.spinner("Analyzing documents & generating response..."):
                    try:
                        # 1. Embed query
                        q_emb = get_embedding(api_key_input, user_query)
                        
                        # 2. Search database
                        retrieved_chunks = vector_store.search(q_emb, top_k=top_k, min_score=min_score)
                        
                        if not retrieved_chunks:
                            response_text = (
                                "No relevant matching passages were found in the database. "
                                "Please make sure the topic is mentioned in your documents, "
                                "or lower the 'Minimum Similarity Match' filter in the sidebar."
                            )
                            message_placeholder.markdown(response_text)
                            st.session_state.messages.append({
                                "role": "assistant",
                                "content": response_text,
                                "sources": []
                            })
                        else:
                            # 3. Generate Answer
                            response_text = generate_answer(
                                api_key=api_key_input,
                                query=user_query,
                                retrieved_chunks=retrieved_chunks,
                                model_name=selected_model,
                                temperature=temperature
                            )
                            
                            # Update UI
                            message_placeholder.markdown(response_text)
                            
                            # Render Sources below
                            with st.expander("🔍 Citations & Retrieved Sources"):
                                for idx, src in enumerate(retrieved_chunks):
                                    st.markdown(f"""
                                    <div class="source-card">
                                        <div class="source-meta">
                                            <span>📄 {src['filename']} ({src['location']})</span>
                                            <span>Score: {src['score']:.2f}</span>
                                        </div>
                                        <div class="source-text">"{src['text']}"</div>
                                    </div>
                                    """, unsafe_allow_html=True)
                            
                            # Append to session state
                            st.session_state.messages.append({
                                "role": "assistant",
                                "content": response_text,
                                "sources": retrieved_chunks
                            })
                    except Exception as e:
                        error_msg = f"❌ **Error occurred:** {str(e)}"
                        message_placeholder.markdown(error_msg)
                        st.session_state.messages.append({"role": "assistant", "content": error_msg})


# ----------------- TAB 2: DOCUMENT LIBRARY -----------------
with tab_library:
    st.markdown("### 📤 Upload New Documents")
    uploaded_files = st.file_uploader(
        "Upload legal documents (PDF, DOCX, or TXT)",
        type=["pdf", "docx", "txt"],
        accept_multiple_files=True
    )
    
    if uploaded_files:
        if not api_key_input or not api_key_input.strip():
            st.warning("⚠️ Please configure your Gemini API Key in the sidebar before uploading and processing documents.")
        else:
            # Show a processing button
            if st.button("Process & Ingest Uploaded Documents", type="primary"):
                for uploaded_file in uploaded_files:
                    with st.spinner(f"Ingesting '{uploaded_file.name}'..."):
                        try:
                            # Create a temporary file to parse
                            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as temp_file:
                                temp_file.write(uploaded_file.read())
                                temp_file_path = temp_file.name

                            try:
                                # 1. Parse document into text chunks
                                raw_chunks = parse_document(temp_file_path, chunk_size, chunk_overlap)
                                
                                if not raw_chunks:
                                    st.error(f"Could not extract any readable text from '{uploaded_file.name}'.")
                                    continue
                                    
                                # 2. Generate embeddings for all chunks in batch
                                chunk_texts = [c["text"] for c in raw_chunks]
                                embeddings = get_embeddings_batch(api_key_input, chunk_texts)
                                
                                # Assign embeddings back to chunks
                                for idx, emb in enumerate(embeddings):
                                    raw_chunks[idx]["embedding"] = emb
                                
                                # 3. Save to Local SQLite Vector Store
                                file_type = os.path.splitext(uploaded_file.name)[1].replace(".", "").upper()
                                vector_store.add_document(
                                    filename=uploaded_file.name,
                                    file_type=file_type,
                                    chunks_data=raw_chunks
                                )
                                st.success(f"Successfully ingested '{uploaded_file.name}' ({len(raw_chunks)} chunks).")
                            finally:
                                # Ensure temp file cleanup
                                if os.path.exists(temp_file_path):
                                    os.remove(temp_file_path)
                                    
                        except ValueError as ve:
                            st.warning(f"ℹ️ {str(ve)}")
                        except Exception as e:
                            st.error(f"❌ Failed to process '{uploaded_file.name}': {str(e)}")
                
                # Rerun to update sidebar stats
                st.rerun()

    st.markdown("---")
    st.markdown("### 📋 Ingested Documents Inventory")
    
    documents_list = vector_store.get_all_documents()
    
    if not documents_list:
        st.info("No documents have been ingested yet.")
    else:
        # Build document data list
        db = vector_store._get_connection()
        doc_stats = []
        try:
            cursor = db.cursor()
            for doc in documents_list:
                # Get chunk count per document
                cursor.execute(
                    "SELECT COUNT(*) FROM chunks JOIN documents ON chunks.doc_id = documents.id WHERE documents.filename = ?",
                    (doc["filename"],)
                )
                chunk_count = cursor.fetchone()[0]
                doc_stats.append({
                    "Document Name": doc["filename"],
                    "File Type": doc["file_type"],
                    "Chunks Count": chunk_count,
                    "Ingested At": doc["upload_date"]
                })
        finally:
            db.close()
            
        df = pd.DataFrame(doc_stats)
        
        # Display table with streamlit dataframe
        st.dataframe(df, use_container_width=True)
        
        # Document Deletion UI
        st.markdown("### 🗑️ Delete Document")
        col_select, col_del = st.columns([3, 1])
        with col_select:
            doc_to_delete = st.selectbox(
                "Select a document to delete",
                options=[doc["filename"] for doc in documents_list],
                label_visibility="collapsed"
            )
        with col_del:
            if st.button("Delete Selected", type="secondary", use_container_width=True):
                if vector_store.delete_document(doc_to_delete):
                    st.success(f"Successfully deleted document '{doc_to_delete}' and all its associated vector embeddings.")
                    st.rerun()
                else:
                    st.error("Failed to delete document.")


# ----------------- TAB 3: VISUAL ANALYTICS -----------------
with tab_analytics:
    st.markdown("### 📊 Document Corpus Analytics Dashboard")
    
    if db_stats["total_documents"] == 0:
        st.info("Ingest documents in the library to view corpus analytics.")
    else:
        # Collate stats
        db = vector_store._get_connection()
        analytics_data = []
        try:
            cursor = db.cursor()
            cursor.execute("""
                SELECT documents.filename, documents.file_type, COUNT(chunks.id) as chunk_count, 
                       SUM(LENGTH(chunks.text)) as char_count
                FROM documents
                LEFT JOIN chunks ON documents.id = chunks.doc_id
                GROUP BY documents.id
            """)
            rows = cursor.fetchall()
            for r in rows:
                analytics_data.append({
                    "filename": r[0],
                    "file_type": r[1],
                    "chunks": r[2] or 0,
                    "characters": r[3] or 0
                })
        finally:
            db.close()
            
        df_an = pd.DataFrame(analytics_data)
        
        # Stat cards
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(f"""
            <div class="stat-card">
                <div class="stat-val">{db_stats['total_documents']}</div>
                <div class="stat-lbl">Active Documents</div>
            </div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown(f"""
            <div class="stat-card">
                <div class="stat-val">{db_stats['total_chunks']}</div>
                <div class="stat-lbl">Total Text Chunks</div>
            </div>
            """, unsafe_allow_html=True)
        with col3:
            total_chars = int(df_an["characters"].sum())
            st.markdown(f"""
            <div class="stat-card">
                <div class="stat-val">{total_chars:,}</div>
                <div class="stat-lbl">Total Characters</div>
            </div>
            """, unsafe_allow_html=True)
            
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Split charts
        chart_col1, chart_col2 = st.columns(2)
        with chart_col1:
            st.markdown("#### 📂 Text Chunks per Document")
            st.bar_chart(
                df_an, 
                x="filename", 
                y="chunks", 
                color="#D4AF37", 
                use_container_width=True
            )
            
        with chart_col2:
            st.markdown("#### 🔠 Text Volume (Characters) per Document")
            st.bar_chart(
                df_an, 
                x="filename", 
                y="characters", 
                color="#8A9A86", 
                use_container_width=True
            )
            
        st.markdown("#### 📁 File Format Distribution")
        format_counts = df_an["file_type"].value_counts().reset_index()
        format_counts.columns = ["Format", "Count"]
        st.dataframe(format_counts, hide_index=True, use_container_width=True)
