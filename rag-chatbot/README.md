## 🤖 RAG Chatbot with Vector Database

A comprehensive Retrieval-Augmented Generation (RAG) system built with LangChain, ChromaDB, and Ollama. This project demonstrates vector database operations and document question-answering capabilities with a beautiful, animated UI.

### ✨ Features

**Week 2 Deliverables Complete:**
- ✅ **Vector Database Setup** (ChromaDB with embeddings)
- ✅ **RAG System** (Document Q&A with source citations)

**Core Functionality:**
- 📤 Upload PDF, TXT, or MD documents
- 🧠 Generate embeddings using sentence-transformers
- 💾 Store vectors in ChromaDB
- 🔍 Semantic search across documents
- 💬 Ask questions and get answers with source citations
- 📊 Similarity score visualizations with Plotly
- 🎨 Beautiful animated UI with Streamlit

**Advanced Features:**
- Document chunking with overlap for better context
- Multiple document upload support
- Real-time vector database statistics
- Configurable LLM models and parameters
- Chat history tracking
- Sample documents included

---

## 🏗️ Project Structure

```
rag-chatbot/
├── app.py                      # Main Streamlit application
├── document_processor.py       # Document loading and chunking
├── vector_db.py                # ChromaDB vector database manager
├── rag_chain.py                # RAG chain implementation
├── requirements.txt            # Python dependencies
├── README.md                   # This file
├── data/                       # Sample documents
│   ├── artificial_intelligence.txt
│   ├── climate_change.txt
│   └── python_programming.txt
└── chroma_db/                  # ChromaDB storage (created at runtime)
```

---

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- Ollama installed and running
- At least one Ollama model pulled (llama3 recommended)

### Installation

1. **Navigate to project directory:**
```bash
cd rag-chatbot
```

2. **Create virtual environment:**
```bash
python3 -m venv venv
source venv/bin/activate
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

4. **Ensure Ollama is running:**
```bash
# In a separate terminal
ollama serve
```

5. **Verify model is available:**
```bash
ollama list
# Make sure llama3 or another model is listed
```

### Running the Application

```bash
streamlit run app.py
```

The application will open in your browser at `http://localhost:8501`

---

## 📖 How to Use

### 1. Upload Documents

**Tab: 📤 Upload Documents**

- Click "Choose files to upload" and select PDF, TXT, or MD files
- Click "🚀 Process & Load Documents" to add them to the vector database
- Or click "📁 Load Sample Documents" to load the included sample files

**What happens:**
- Documents are split into chunks (1000 characters with 200 overlap)
- Embeddings are generated using `all-MiniLM-L6-v2`
- Vectors are stored in ChromaDB
- Progress is shown with animated progress bars

### 2. Ask Questions

**Tab: 💬 Ask Questions**

- Enter your question in the text box
- Click "🔍 Get Answer"
- View the answer, similarity scores chart, and source documents

**Example Questions:**
```
What is machine learning?
Explain climate change impacts on agriculture
What are Python decorators?
How does deep learning work?
```

**The system will:**
- Retrieve the most relevant document chunks
- Show similarity scores as a bar chart
- Generate an answer using Ollama LLM
- Display source documents with scores
- Track conversation history

### 3. Explore Database

**Tab: 🔍 Explore Database**

- View vector database statistics
- Perform direct semantic searches
- Visualize similarity scores
- Inspect retrieved chunks

---

## 🎨 UI Features

### Animations & Visualizations

1. **Progress Indicators**
   - Animated progress bars during document processing
   - Loading spinners for LLM generation
   - Success animations (balloons) after uploads

2. **Charts & Graphs**
   - Plotly bar charts for similarity scores
   - Color-coded scores (Viridis colorscale)
   - Interactive hover information

3. **Modern Design**
   - Gradient headers and metric cards
   - Expandable source cards
   - Tab-based navigation
   - Responsive layout

### Sidebar Configuration

- **Vector DB Stats:** Real-time document count and status
- **Model Settings:** Choose LLM model and temperature
- **Number of Sources:** Adjust retrieval count (1-10)
- **Clear Database:** Reset the vector database
- **Loaded Documents:** View uploaded files

---

## 🔧 Technical Details

### Architecture

```
User Query
    ↓
Semantic Search (ChromaDB)
    ↓
Retrieve Top-K Documents
    ↓
Format Context
    ↓
LLM Generation (Ollama)
    ↓
Answer + Sources
```

### Components

**1. Document Processor** (`document_processor.py`)
- Loads PDF/TXT/MD files
- Splits text using `RecursiveCharacterTextSplitter`
- Chunk size: 1000 characters
- Overlap: 200 characters
- Provides chunking statistics

**2. Vector Database Manager** (`vector_db.py`)
- ChromaDB integration via LangChain
- Embedding model: `all-MiniLM-L6-v2` (384 dimensions)
- Persistent storage in `./chroma_db`
- Similarity search with scores
- Collection management

**3. RAG Chain** (`rag_chain.py`)
- LCEL (LangChain Expression Language) chain
- Ollama LLM integration
- Retriever with configurable K
- Prompt template for RAG
- Source tracking and formatting

**4. Streamlit UI** (`app.py`)
- Three-tab interface
- Real-time database stats
- Plotly visualizations
- Custom CSS styling
- Session state management

### Embedding Model

**all-MiniLM-L6-v2:**
- Size: ~90MB
- Embedding dimension: 384
- Fast inference on CPU
- Good quality for semantic search
- Downloaded automatically on first run

### Chunking Strategy

**RecursiveCharacterTextSplitter:**
- Splits by: `\n\n`, `\n`, ` `, then character
- Preserves paragraph structure
- Overlap maintains context between chunks
- Configurable size and overlap

---

## ⚙️ Configuration

### Model Settings

Available in sidebar:
- **LLM Model:** llama3, mistral, llama3.2, gemma2
- **Temperature:** 0.0 (focused) to 1.0 (creative)
- **Number of Sources:** 1-10 chunks to retrieve

### Advanced Configuration

Edit these in the code:

**Document Processing:**
```python
# document_processor.py
chunk_size = 1000       # Chunk size in characters
chunk_overlap = 200     # Overlap between chunks
```

**Vector Database:**
```python
# vector_db.py
model_name = "all-MiniLM-L6-v2"  # Embedding model
persist_directory = "./chroma_db"  # Storage location
```

**RAG Chain:**
```python
# rag_chain.py
temperature = 0.7  # LLM creativity
k = 4              # Number of chunks to retrieve
```

---

## 📊 Deliverables Checklist

### ✅ Deliverable #2: Vector Database Setup

- [x] ChromaDB installed and configured
- [x] Sentence-transformers for embeddings
- [x] Document loading and chunking
- [x] Vector storage and retrieval
- [x] Similarity search functionality
- [x] Sample data included
- [x] Database statistics display

### ✅ Deliverable #3: RAG System

- [x] Document Q&A functionality
- [x] Retrieval-augmented generation
- [x] Source citations displayed
- [x] LangChain LCEL chains
- [x] Ollama integration
- [x] Context-aware answers
- [x] Chat history tracking

### ✅ Bonus: Enhanced UI/UX

- [x] Animations and progress indicators
- [x] Similarity score visualizations
- [x] Tab-based interface
- [x] Real-time statistics
- [x] Modern design with custom CSS
- [x] Interactive charts (Plotly)

---

## 🧪 Testing

### Test with Sample Documents

1. Load sample documents (included in `data/` folder)
2. Ask these questions:

```
# AI Document
- "What is machine learning?"
- "Explain deep learning architectures"
- "What are ethical considerations in AI?"

# Climate Change Document
- "What causes climate change?"
- "What are the impacts of rising sea levels?"
- "How can we mitigate climate change?"

# Python Document
- "What is Python used for?"
- "Explain Python decorators"
- "What are popular Python libraries?"
```

### Test with Your Own Documents

1. Upload your PDF or TXT files
2. Wait for processing
3. Ask domain-specific questions
4. Verify source citations match your documents

---

## 🐛 Troubleshooting

### Ollama Not Running

**Error:** "❌ Ollama is not running!"

**Solution:**
```bash
ollama serve
```

### Model Not Found

**Error:** Model errors during generation

**Solution:**
```bash
ollama pull llama3
ollama list  # Verify it's installed
```

### Import Errors

**Error:** ModuleNotFoundError

**Solution:**
```bash
pip install -r requirements.txt
```

### ChromaDB Errors

**Error:** Collection issues

**Solution:**
```bash
# Delete the chroma_db folder
rm -rf chroma_db
# Restart the app
streamlit run app.py
```

### Slow Performance

**Issue:** Slow document processing or query responses

**Solutions:**
- Use smaller documents or fewer chunks
- Reduce number of sources (K) in sidebar
- Use faster Ollama model (llama3.2 instead of llama3)
- Ensure you have enough RAM

---

## 📚 Learning Resources

### Concepts Covered

- **Vector Databases:** Storing and searching embeddings
- **Embeddings:** Converting text to numerical vectors
- **Semantic Search:** Finding similar documents by meaning
- **RAG:** Combining retrieval with generation
- **LangChain LCEL:** Modern chain composition
- **Document Chunking:** Splitting for optimal retrieval

### Related Documentation

- [ChromaDB Docs](https://docs.trychroma.com/)
- [LangChain Documentation](https://python.langchain.com/)
- [Sentence Transformers](https://www.sbert.net/)
- [Ollama Models](https://ollama.com/library)
- [Streamlit Docs](https://docs.streamlit.io/)

---

## 🚀 Future Enhancements

Possible improvements:
- [ ] Add Milvus for comparison
- [ ] Support more file types (DOCX, PPTX)
- [ ] Implement conversation memory in RAG
- [ ] Add re-ranking for better retrieval
- [ ] Support multiple collections
- [ ] Implement hybrid search (keyword + semantic)
- [ ] Add user authentication
- [ ] Deploy to cloud (Streamlit Cloud, AWS, etc.)

---

## 📄 License

MIT License - Feel free to use for learning and projects!

---

## 🙏 Acknowledgments

- Built as part of Week 2 LangChain & RAG learning
- Uses open-source models from Ollama and HuggingFace
- Inspired by modern RAG best practices

---

## 📧 Support

If you encounter issues:
1. Check the troubleshooting section
2. Verify all prerequisites are installed
3. Ensure Ollama is running with a model loaded
4. Check terminal for error messages

---

**Happy RAG Building! 🎉**
