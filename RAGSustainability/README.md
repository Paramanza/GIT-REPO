# 🌱 RAG Sustainability Chatbot

A Retrieval Augmented Generation (RAG) chatbot focused on sustainability topics, featuring real-time 3D vector space visualization and interactive knowledge exploration.

## 📋 Features

- **Interactive Chat Interface**: Gradio-based web UI with conversation memory
- **3D Vector Visualization**: Real-time Plotly visualization of document embeddings
- **Smart Document Retrieval**: Contextual chunk retrieval with highlighting
- **Sustainability Knowledge Base**: Comprehensive collection from 40+ organizations
- **Clean Architecture**: Modular, maintainable codebase with clear separation of concerns

## 🏗️ Architecture

The application follows a clean, modular design:

```
├── app.py                    # Main Gradio application and UI orchestration
├── chunker.py               # Document loading and text chunking pipeline
├── vector_store.py          # Vector database management and 3D visualization
├── rag_chain.py            # RAG conversation logic and response formatting
├── build_database.py       # Utility script to build/rebuild vector database
├── knowledge-base/         # Source documents organized by organization
├── vector_db/              # Chroma vector database (generated)
├── sample_questions.txt    # Curated test questions for system evaluation
├── requirements.txt        # Python dependencies
├── Dockerfile             # Production container configuration
└── fly.toml               # Fly.io deployment configuration
```

### Module Responsibilities

| Module | Purpose | Key Classes |
|--------|---------|-------------|
| **chunker.py** | Document processing pipeline | `DocumentChunker`, `SmartTextLoader` |
| **vector_store.py** | Vector database & 3D visualization | `VectorStoreManager` |
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

```bash
# Build the vector database from knowledge base documents
python build_database.py
```

This will:
- Process all documents in the `knowledge-base/` directory
- Create optimally-sized chunks with metadata
- Generate embeddings and store in Chroma database
- Create a preview file for inspection

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

```bash
python build_database.py
```

### Customizing Parameters

Key configuration in `app.py`:

```python
MODEL = "gpt-4o-mini"          # OpenAI model
K_FACTOR = 25                  # Chunks retrieved per query  
TEMPERATURE = 0.7              # Response creativity (0-1)
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

## 🐳 Docker Deployment

### Quick Deploy to Fly.io

```bash
# 1. Install Fly.io CLI
curl -L https://fly.io/install.sh | sh

# 2. Login to Fly.io
flyctl auth login

# 3. Update app name in fly.toml (make it unique)
# Edit the app name to something unique for your deployment

# 4. Set your OpenAI API key
flyctl secrets set OPENAI_API_KEY=your_actual_openai_api_key_here

# 5. Deploy
flyctl deploy

# 6. Open your deployed app
flyctl open
```

### Local Docker Testing

```bash
# Build the Docker image
docker build -t rag-sustainability-chatbot .

# Run locally (replace with your API key)
docker run -p 7860:7860 -e OPENAI_API_KEY=your_key_here rag-sustainability-chatbot

# Access at http://localhost:7860
```

### Docker Configuration

The deployment includes:

- **Base Image**: `python:3.11-slim` for optimal size/performance
- **Vector Store**: ~500MB included in image for simplicity
- **Port**: Exposes 7860 (Gradio default)
- **Security**: Non-root user, health checks, secret management
- **Optimization**: Layer caching, minimal dependencies

### Resource Requirements

- **Memory**: 4GB recommended (for vector store + ML operations)
- **CPU**: 2 vCPU (shared) for responsive performance
- **Storage**: ~1GB (vector store + dependencies)
- **Cold Start**: ~30-60 seconds for first request

### Secrets Management

Never include secrets in Docker images. Set them via Fly.io:

```bash
# Set OpenAI API key
flyctl secrets set OPENAI_API_KEY=sk-your-key-here

# View current secrets
flyctl secrets list

# Remove a secret
flyctl secrets unset SECRET_NAME
```

### Monitoring & Debugging

```bash
# View logs
flyctl logs

# Check status
flyctl status

# Access app shell
flyctl ssh console

# Scale resources
flyctl scale memory 8gb
flyctl scale count 2
```

#### Common Issues & Solutions

| Issue | Solution |
|-------|----------|
| "Not listening on expected address" | App not binding to 0.0.0.0:7860 - check Gradio startup logs |
| Health check timeouts | Increase grace period in fly.toml |
| Out of memory errors | Increase memory allocation in fly.toml |
| Slow startup (>3 minutes) | Vector store loading - normal for first cold start |
| Docker build fails locally | Ensure vector_db directory exists and has content |

### Cost Optimization

- **Auto-scaling**: Machines stop when idle, start on demand
- **Shared CPU**: More cost-effective than dedicated
- **Single Region**: Deploy to one region initially
- **Monitoring**: Use `flyctl status` to monitor usage

## 📈 Performance

- **Vector Database**: ~3,000 chunks from 500+ documents
- **Response Time**: ~2-3 seconds for complex queries
- **Memory Usage**: ~500MB with full knowledge base loaded
- **Visualization**: Real-time updates with smooth interactions

## 🤝 Contributing

The codebase follows clean architecture principles:

1. **Separation of Concerns**: Each module has single responsibility
2. **Type Hints**: Full typing for better IDE support  
3. **Documentation**: Comprehensive docstrings and comments
4. **Error Handling**: Graceful failure with user feedback

## 📄 License

This project contains sustainability policy documents and organizational information compiled for educational and research purposes. Please respect the original sources and their licensing terms.

---

**Ready to explore sustainable fashion? Launch the app and start asking questions!** 🚀
