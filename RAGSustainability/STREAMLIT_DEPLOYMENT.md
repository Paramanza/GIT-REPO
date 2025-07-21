# Streamlit Cloud Deployment Guide

This guide helps you deploy the RAG Sustainability Chatbot to Streamlit Cloud while handling the SQLite version issue.

## 🚀 Quick Deployment Steps

### Method 1: ChromaDB 0.4.22 (Try This First)

1. **Use updated requirements**: The `requirements.txt` now uses `chromadb==0.4.22`
2. **Push to GitHub**: Commit and push all files
3. **Deploy on Streamlit Cloud**: 
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Connect your GitHub repository
   - Set main file to `streamlit_app.py`
   - Deploy!

### Method 2: FAISS Alternative (If ChromaDB Fails)

1. **Convert database locally**:
   ```bash
   python convert_chroma_to_faiss.py
   ```
2. **Switch files**:
   - Rename `requirements_faiss.txt` → `requirements.txt`
   - Set main file to `streamlit_app_faiss.py`
3. **Commit FAISS files**: Add `faiss_db.*` files to git
4. **Deploy**: Same Streamlit Cloud process

## 🔧 SQLite Issue Resolution

The main issue you encountered was: `RuntimeError: Your system has an unsupported version of sqlite3. Chroma requires sqlite3 >= 3.35.0.`

### Multiple Solutions Available

Choose the approach that works best for your deployment:

#### Option 1: Older ChromaDB Version (Recommended)
- ✅ **File**: `requirements.txt` (updated)
- ✅ **App**: Use `streamlit_app.py`
- Uses `chromadb==0.4.22` which has better Streamlit Cloud compatibility

#### Option 2: FAISS Alternative (Most Reliable)
- ✅ **File**: `requirements_faiss.txt`
- ✅ **App**: Use `streamlit_app_faiss.py`  
- ✅ **Converter**: Run `python convert_chroma_to_faiss.py` locally first
- No SQLite dependency at all - uses FAISS for vector storage

#### Option 3: pysqlite3 Fix (Backup)
- ✅ **File**: `requirements_backup.txt`
- Includes pysqlite3-binary and SQLite module replacement

### Files Added/Modified for SQLite Fix

- ✅ `requirements.txt` - Added `pysqlite3-binary>=0.5.0`
- ✅ `pyproject.toml` - Added streamlit and pysqlite3-binary dependencies
- ✅ `streamlit_app.py` - New Streamlit version with SQLite handling
- ✅ `packages.txt` - System-level dependencies for Streamlit Cloud
- ✅ `.streamlit/config.toml` - Streamlit configuration

### Key Code Changes

The `streamlit_app.py` file includes this critical SQLite fix at the top:

```python
# Handle SQLite version issue for Streamlit Cloud
import sys
import sqlite3
try:
    # Try to use pysqlite3 if available (newer SQLite version)
    import pysqlite3
    sys.modules['sqlite3'] = pysqlite3
except ImportError:
    # Fall back to system sqlite3
    pass
```

## 🔑 Environment Variables

Set up your OpenAI API key in Streamlit Cloud:

1. Go to your app settings in Streamlit Cloud
2. Add a new secret:
   ```toml
   OPENAI_API_KEY = "your-openai-api-key-here"
   ```

## 📁 Required Files Structure

```
your-repo/
├── streamlit_app.py          # Main Streamlit application
├── requirements.txt          # Python dependencies
├── packages.txt             # System dependencies
├── .streamlit/
│   └── config.toml          # Streamlit configuration
├── vector_db/               # Your Chroma database directory
├── knowledge-base/          # Your document collection
└── pyproject.toml           # Project configuration
```

## 🐛 Troubleshooting

### If SQLite Error Persists

1. **Check deployment logs** in Streamlit Cloud for specific error messages
2. **Verify pysqlite3-binary installation** in the logs
3. **Try alternative deployment methods**:
   
   **Option A: Force pysqlite3 installation**
   Add to `requirements.txt`:
   ```
   pysqlite3-binary==0.5.2
   ```

   **Option B: Use chromadb with specific version**
   ```
   chromadb==0.4.22
   ```

### Alternative Vector Database Options

If SQLite issues persist, consider switching to:

1. **Pinecone** (cloud-hosted)
2. **Weaviate** (cloud or self-hosted)
3. **FAISS** (file-based, no SQLite dependency)

### Memory Issues

If you encounter memory issues on Streamlit Cloud:

1. Reduce `K_FACTOR` from 25 to 10-15
2. Implement document chunking for large knowledge bases
3. Use `@st.cache_resource` for expensive operations (already implemented)

## 📊 Performance Optimization

The Streamlit version includes several optimizations:

- **Caching**: RAG system initialization is cached
- **Efficient plotting**: Only updates visualization when needed
- **Chunked display**: Shows only first 5 source documents
- **Responsive layout**: Adapts to different screen sizes

## 🔄 From Gradio to Streamlit

Key differences in the Streamlit version:

- **Chat interface**: Native Streamlit chat components
- **State management**: Uses `st.session_state` for persistence
- **Layout**: Two-column layout with sidebar
- **Interactivity**: Sample questions in sidebar
- **Error handling**: Better user feedback

## 🆘 Support

If you continue to experience issues:

1. Check the [Streamlit Community Forum](https://discuss.streamlit.io/)
2. Review [Chroma troubleshooting guide](https://docs.trychroma.com/troubleshooting#sqlite)
3. Consider filing an issue in your repository with deployment logs

## ✅ Deployment Checklist

Before deploying, ensure:

- [ ] All files are committed to GitHub
- [ ] `OPENAI_API_KEY` is set in Streamlit Cloud secrets
- [ ] `vector_db/` directory exists and contains your Chroma database
- [ ] Main file is set to `streamlit_app.py` in deployment settings
- [ ] Repository is public or you have Streamlit Cloud Pro for private repos

## 🎯 Testing Locally

Test the Streamlit version locally before deploying:

```bash
# Install dependencies
pip install -r requirements.txt

# Run the Streamlit app
streamlit run streamlit_app.py
```

The app should open at `http://localhost:8501` 