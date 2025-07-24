# 🌱 RAG Sustainability Chatbot

A Retrieval Augmented Generation (RAG) chatbot focused on sustainability topics, featuring real-time 3D vector space visualization and interactive knowledge exploration.

## 📋 Features

- **Interactive Chat Interface**: Gradio-based web UI with conversation memory
- **3D Vector Visualization**: Real-time Plotly visualization of document embeddings
- **Smart Document Retrieval**: Contextual chunk retrieval with highlighting
- **Sustainability Knowledge Base**: Comprehensive collection from 40+ organizations
- **Modular Architecture**: Clean, maintainable codebase with separate concerns

## 🏗️ Architecture

The codebase follows a modular design pattern:

```
├── app.py                    # Main Gradio application orchestrator
├── chunker.py               # Document loading and text chunking
├── vector_store.py          # Vector database management and visualization
├── rag_chain.py            # LangChain conversation logic and memory
├── knowledge-base/         # Source documents organized by organization
├── vector_db/              # Chroma vector database (generated)
└── requirements.txt        # Python dependencies
```

### Module Responsibilities

| Module | Purpose | Key Classes |
|--------|---------|-------------|
| **chunker.py** | Document processing pipeline | `DocumentChunker`, `SmartTextLoader` |
| **vector_store.py** | Vector database & visualization | `VectorStoreManager` |
| **rag_chain.py** | Conversation & response handling | `RAGConversationManager`, `RAGResponseProcessor` |
| **app.py** | UI orchestration & main entry | Gradio interface functions |

## 🚀 Quick Start

### 1. Installation

```bash
# Clone and navigate to repository
cd RAGSustainability

# Install dependencies
pip install -r requirements.txt
```

### 2. Environment Setup

Create a `.env` file:
```env
OPENAI_API_KEY=your_openai_api_key_here
```

### 3. Build Vector Database

If no vector database exists, create one from the knowledge base:

```python
# Run this script to process documents and create vector store
from chunker import create_chunks_for_vectorstore
from vector_store import create_vector_store_from_chunks
from langchain_openai import OpenAIEmbeddings

# Process documents
chunks = create_chunks_for_vectorstore(create_preview=True)

# Create vector store
embeddings = OpenAIEmbeddings()
vector_store_manager = create_vector_store_from_chunks(
    chunks=chunks,
    embeddings=embeddings
)
```

### 4. Launch Application

```bash
python app.py
```

The application will open in your browser with:
- Chat interface on the left
- 3D vector visualization on the right
- Retrieved knowledge chunks displayed below chat

## 📊 Knowledge Base

The system contains sustainability documents from 40+ organizations:

| Category | Organizations | Example Topics |
|----------|--------------|----------------|
| **Policy & Regulation** | EU Parliament, UK Gov, US Congress | Green claims, circular economy, EPR |
| **Certifications** | GOTS, Cradle to Cradle, OEKO-TEX | Textile standards, material health |
| **NGOs & Advocacy** | Fashion Revolution, Clean Clothes | Labor rights, transparency |
| **Industry Standards** | Better Cotton, Fair Trade | Supply chain, working conditions |

### Document Processing

- **Chunk Size**: 1,600 tokens with 400 token overlap
- **Smart Encoding**: Automatic detection for international documents  
- **Metadata Enrichment**: Organization tagging for source attribution
- **Semantic Splitting**: Prioritizes paragraph and sentence boundaries

## 🎯 Usage Examples

### Sample Questions

The system excels at cross-document analysis and policy comparison:

**Cross-Jurisdictional Comparison:**
> "What are the key differences between the UK's Green Claims Code and the US FTC Green Guides regarding eco-friendly marketing?"

**Impact Analysis:**
> "How much water is required to produce a cotton T-shirt, and how does this relate to EU circular economy goals?"

**Practical Application:**
> "What information must a fashion brand provide to comply with both EU Digital Product Passport requirements and FTC Green Guides?"

### 3D Visualization Features

- **Color Coding**: Each organization type has distinct colors
- **Query Highlighting**: Retrieved chunks appear in white
- **Query Position**: Your question shown as yellow diamond
- **Interactive Exploration**: Hover to see chunk previews

## 🔧 Development

### Adding New Documents

1. Create organization folder in `knowledge-base/`
2. Add `.txt` files with relevant content
3. Rebuild vector database:

```python
from chunker import create_chunks_for_vectorstore
from vector_store import create_vector_store_from_chunks

chunks = create_chunks_for_vectorstore()
# Vector store will auto-update with new content
```

### Customizing Parameters

Key configuration in `app.py`:

```python
MODEL = "gpt-4o-mini"          # OpenAI model
K_FACTOR = 25                  # Chunks retrieved per query  
TEMPERATURE = 0.7              # Response creativity
CHUNK_SIZE = 1600              # Token size per chunk
CHUNK_OVERLAP = 400            # Overlap between chunks
```

### Extending Functionality

The modular design supports easy extension:

- **New Visualizations**: Extend `VectorStoreManager` 
- **Different UIs**: Create new interface using existing modules
- **Custom Processing**: Inherit from `DocumentChunker`
- **Enhanced Memory**: Extend `RAGConversationManager`

## 📦 Dependencies

Core libraries and their purposes:

| Library | Purpose |
|---------|---------|
| **gradio** | Web interface and interactions |
| **langchain** | RAG pipeline and conversation memory |
| **chromadb** | Vector database for embeddings |
| **plotly** | 3D visualization and interactivity |
| **scikit-learn** | PCA dimensionality reduction |
| **openai** | Text embeddings and chat completions |

## 🐳 Deployment

The codebase is ready for containerization:

1. **Clean Dependencies**: No unused packages
2. **Modular Structure**: Easy to package
3. **Environment Variables**: Configurable API keys
4. **Error Handling**: Graceful failure modes

## 📈 Performance

- **Vector Database**: ~3,000 chunks from 500+ documents
- **Response Time**: ~2-3 seconds for complex queries
- **Memory Usage**: ~500MB with full knowledge base loaded
- **Visualization**: Real-time updates with 60fps interactions

## 🤝 Contributing

The codebase follows clean architecture principles:

1. **Separation of Concerns**: Each module has single responsibility
2. **Type Hints**: Full typing for better IDE support  
3. **Documentation**: Comprehensive docstrings
4. **Error Handling**: Graceful failure with user feedback

## 📄 License

This project contains sustainability policy documents and organizational information compiled for educational and research purposes. Please respect the original sources and their licensing terms.

---

**Ready to explore sustainable fashion? Launch the app and start asking questions!** 🚀
