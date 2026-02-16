"""
RAG Chatbot with ChromaDB - Streamlit Application
Week 2 Deliverables: Vector Database + RAG System
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from document_processor import DocumentProcessor
from vector_db import VectorDBManager
from rag_chain import RAGChain
import os
import time
import requests

# Page configuration
st.set_page_config(
    page_title="RAG Chatbot with Vector DB",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        padding: 1rem 0;
    }

    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }

    .source-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #667eea;
        margin: 0.5rem 0;
    }

    .stButton>button {
        width: 100%;
        border-radius: 8px;
        height: 3rem;
        font-weight: 600;
        transition: all 0.3s;
    }

    .success-message {
        padding: 1rem;
        background: #d4edda;
        border-left: 4px solid #28a745;
        border-radius: 4px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)


# Initialize session state
def init_session_state():
    """Initialize session state variables"""
    if 'vector_db' not in st.session_state:
        st.session_state.vector_db = None
    if 'rag_chain' not in st.session_state:
        st.session_state.rag_chain = None
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'documents_loaded' not in st.session_state:
        st.session_state.documents_loaded = []
    if 'db_initialized' not in st.session_state:
        st.session_state.db_initialized = False


def check_ollama_connection():
    """Check if Ollama is running"""
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        return response.status_code == 200
    except:
        return False


def initialize_components():
    """Initialize vector DB and RAG chain"""
    if not st.session_state.db_initialized:
        with st.spinner("🔧 Initializing Vector Database..."):
            st.session_state.vector_db = VectorDBManager(
                persist_directory="./chroma_db",
                collection_name="rag_documents"
            )
            st.session_state.rag_chain = RAGChain(
                vector_db_manager=st.session_state.vector_db,
                model_name="llama3"
            )
            st.session_state.db_initialized = True
            time.sleep(0.5)


def create_similarity_chart(sources, scores):
    """Create similarity score visualization"""
    if not sources or not scores:
        return None

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=[f"Source {i+1}" for i in range(len(scores))],
        y=scores,
        marker=dict(
            color=scores,
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(title="Similarity")
        ),
        text=[f"{score:.2%}" for score in scores],
        textposition='outside'
    ))

    fig.update_layout(
        title="📊 Document Similarity Scores",
        xaxis_title="Retrieved Sources",
        yaxis_title="Similarity Score",
        yaxis=dict(range=[0, 1]),
        height=400,
        template="plotly_white"
    )

    return fig


def process_and_load_documents(uploaded_files):
    """Process and load documents into vector database"""
    processor = DocumentProcessor(chunk_size=1000, chunk_overlap=200)

    progress_bar = st.progress(0)
    status_text = st.empty()

    total_chunks = 0
    all_chunks = []

    for idx, uploaded_file in enumerate(uploaded_files):
        # Update progress
        progress = (idx + 1) / len(uploaded_files)
        progress_bar.progress(progress)
        status_text.text(f"📄 Processing {uploaded_file.name}...")

        # Save uploaded file temporarily
        temp_path = f"./temp_{uploaded_file.name}"
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        try:
            # Process document
            result = processor.process_document(temp_path)

            # Add to collection
            all_chunks.extend(result['chunks'])
            total_chunks += result['num_chunks']

            # Store filename
            st.session_state.documents_loaded.append(uploaded_file.name)

            # Clean up temp file
            os.remove(temp_path)

        except Exception as e:
            st.error(f"Error processing {uploaded_file.name}: {str(e)}")
            if os.path.exists(temp_path):
                os.remove(temp_path)

    # Add all chunks to vector database
    if all_chunks:
        status_text.text("💾 Adding documents to vector database...")
        result = st.session_state.vector_db.add_documents(all_chunks)

        if result['status'] == 'success':
            progress_bar.progress(1.0)
            status_text.empty()
            return True, total_chunks
        else:
            st.error(result['message'])
            return False, 0

    return False, 0


def main():
    """Main application"""

    # Initialize
    init_session_state()
    initialize_components()

    # Header
    st.markdown('<h1 class="main-header">🤖 RAG Chatbot with Vector Database</h1>', unsafe_allow_html=True)
    st.markdown("**Week 2 Deliverables:** ChromaDB Setup + RAG System with Document Q&A")

    # Check Ollama connection
    if not check_ollama_connection():
        st.error("❌ **Ollama is not running!** Please start Ollama to use the RAG chatbot.")
        st.info("Run: `ollama serve` in your terminal")
        st.stop()

    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")

        # Vector DB Stats
        st.subheader("📊 Vector Database Stats")
        stats = st.session_state.vector_db.get_collection_stats()

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Documents", stats.get('total_documents', 0))
        with col2:
            st.metric("Dimension", stats.get('embedding_dimension', 384))

        st.caption(f"Status: {stats.get('status', 'Unknown').upper()}")

        st.divider()

        # Model Settings
        st.subheader("🎛️ Model Settings")

        model = st.selectbox(
            "LLM Model",
            ["llama3", "mistral", "llama3.2", "gemma2"],
            index=0
        )

        temperature = st.slider(
            "Temperature",
            min_value=0.0,
            max_value=1.0,
            value=0.7,
            step=0.1,
            help="Higher = more creative, Lower = more focused"
        )

        num_sources = st.slider(
            "Number of Sources",
            min_value=1,
            max_value=10,
            value=4,
            help="Number of document chunks to retrieve"
        )

        # Update model if changed
        if model != st.session_state.rag_chain.model_name:
            st.session_state.rag_chain.update_model(model, temperature)

        st.divider()

        # Clear database
        if st.button("🗑️ Clear Database", type="secondary"):
            result = st.session_state.vector_db.delete_collection()
            if result['status'] == 'success':
                st.session_state.documents_loaded = []
                st.success("Database cleared!")
                st.rerun()

        # Loaded documents
        if st.session_state.documents_loaded:
            st.subheader("📚 Loaded Documents")
            for doc in st.session_state.documents_loaded:
                st.caption(f"✓ {doc}")

    # Main tabs
    tab1, tab2, tab3 = st.tabs(["📤 Upload Documents", "💬 Ask Questions", "🔍 Explore Database"])

    # Tab 1: Upload Documents
    with tab1:
        st.header("📤 Upload Documents to Vector Database")

        st.info("📝 **Supported formats:** PDF, TXT, MD | **Sample documents** are available in the `data/` folder")

        uploaded_files = st.file_uploader(
            "Choose files to upload",
            type=['pdf', 'txt', 'md'],
            accept_multiple_files=True
        )

        col1, col2 = st.columns([2, 1])

        with col1:
            if st.button("🚀 Process & Load Documents", disabled=not uploaded_files, type="primary"):
                if uploaded_files:
                    with st.spinner("Processing documents..."):
                        success, total_chunks = process_and_load_documents(uploaded_files)

                        if success:
                            st.balloons()
                            st.success(f"✅ Successfully loaded {len(uploaded_files)} documents with {total_chunks} chunks!")

                            # Show updated stats
                            stats = st.session_state.vector_db.get_collection_stats()
                            col_a, col_b, col_c = st.columns(3)
                            with col_a:
                                st.metric("Total Documents in DB", stats.get('total_documents', 0))
                            with col_b:
                                st.metric("Files Just Added", len(uploaded_files))
                            with col_c:
                                st.metric("Chunks Created", total_chunks)

        with col2:
            # Load sample documents
            if st.button("📁 Load Sample Documents", type="secondary"):
                sample_dir = "./data"
                if os.path.exists(sample_dir):
                    sample_files = [f for f in os.listdir(sample_dir) if f.endswith(('.txt', '.pdf', '.md'))]

                    if sample_files:
                        processor = DocumentProcessor()
                        progress_bar = st.progress(0)

                        all_chunks = []
                        for idx, filename in enumerate(sample_files):
                            filepath = os.path.join(sample_dir, filename)
                            result = processor.process_document(filepath)
                            all_chunks.extend(result['chunks'])

                            progress_bar.progress((idx + 1) / len(sample_files))

                        st.session_state.vector_db.add_documents(all_chunks)
                        st.session_state.documents_loaded.extend(sample_files)

                        st.success(f"✅ Loaded {len(sample_files)} sample documents!")
                        st.rerun()
                    else:
                        st.warning("No sample documents found in data/ folder")
                else:
                    st.warning("data/ folder not found")

    # Tab 2: Ask Questions
    with tab2:
        st.header("💬 Ask Questions About Your Documents")

        # Check if database has documents
        stats = st.session_state.vector_db.get_collection_stats()
        if stats.get('total_documents', 0) == 0:
            st.warning("⚠️ No documents in the database. Please upload documents first!")
            st.stop()

        # Question input
        question = st.text_input(
            "Enter your question:",
            placeholder="e.g., What is machine learning?",
            key="question_input"
        )

        if st.button("🔍 Get Answer", type="primary", disabled=not question):
            if question:
                with st.spinner("🤔 Thinking..."):
                    # Get answer with scores
                    result = st.session_state.rag_chain.ask_with_scores(
                        question,
                        k=num_sources
                    )

                    if result['status'] == 'success':
                        # Display answer
                        st.markdown("### 🎯 Answer")
                        st.markdown(f"**Q:** {question}")
                        st.markdown(result['answer'])

                        # Display similarity scores chart
                        if result['scores']:
                            st.plotly_chart(
                                create_similarity_chart(result['sources'], result['scores']),
                                use_container_width=True
                            )

                        # Display sources
                        st.markdown("### 📚 Sources")
                        st.caption(f"Retrieved {result['num_sources']} relevant document chunks")

                        for idx, source in enumerate(result['sources']):
                            with st.expander(f"📄 Source {idx + 1} (Similarity: {source['similarity_score']:.2%})"):
                                st.markdown(f"**Content:**")
                                st.text(source['full_content'])

                                if source.get('metadata'):
                                    st.markdown("**Metadata:**")
                                    st.json(source['metadata'])

                        # Add to chat history
                        st.session_state.chat_history.append({
                            "question": question,
                            "answer": result['answer'],
                            "num_sources": result['num_sources']
                        })

                    else:
                        st.error(f"Error: {result['answer']}")

        # Chat history
        if st.session_state.chat_history:
            st.markdown("---")
            st.subheader("📜 Chat History")

            for idx, chat in enumerate(reversed(st.session_state.chat_history[-5:])):
                with st.expander(f"Q{len(st.session_state.chat_history) - idx}: {chat['question'][:60]}..."):
                    st.markdown(f"**Question:** {chat['question']}")
                    st.markdown(f"**Answer:** {chat['answer']}")
                    st.caption(f"Sources used: {chat['num_sources']}")

    # Tab 3: Explore Database
    with tab3:
        st.header("🔍 Explore Vector Database")

        # Database stats
        stats = st.session_state.vector_db.get_collection_stats()

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("📊 Total Documents", stats.get('total_documents', 0))
        with col2:
            st.metric("🎯 Embedding Dimension", stats.get('embedding_dimension', 384))
        with col3:
            st.metric("📁 Files Loaded", len(st.session_state.documents_loaded))

        st.divider()

        # Semantic search
        st.subheader("🔎 Semantic Search")
        st.caption("Search the vector database directly without generating an answer")

        search_query = st.text_input(
            "Enter search query:",
            placeholder="e.g., neural networks",
            key="search_query"
        )

        if st.button("Search", disabled=not search_query):
            if search_query:
                with st.spinner("Searching..."):
                    results = st.session_state.vector_db.similarity_search(
                        search_query,
                        k=num_sources
                    )

                    if results:
                        st.success(f"Found {len(results)} relevant chunks")

                        # Create visualization
                        scores = [score for _, score, _ in results]
                        fig = create_similarity_chart(results, scores)
                        if fig:
                            st.plotly_chart(fig, use_container_width=True)

                        # Display results
                        for idx, (content, score, metadata) in enumerate(results):
                            with st.expander(f"Result {idx + 1} (Score: {score:.2%})"):
                                st.markdown(content)
                                if metadata:
                                    st.caption(f"Metadata: {metadata}")
                    else:
                        st.warning("No results found")

    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: #666;'>
            <p><strong>RAG Chatbot with ChromaDB</strong> |
            Week 2 Deliverables: Vector Database + RAG System |
            Built with LangChain, Ollama & Streamlit</p>
        </div>
        """,
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()
