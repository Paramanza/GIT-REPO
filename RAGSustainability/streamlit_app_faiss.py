#!/usr/bin/env python3
"""
FAISS-based Streamlit version of the RAG Sustainability Chatbot
Uses FAISS instead of Chroma to avoid SQLite issues on Streamlit Cloud
"""

import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from dotenv import load_dotenv
import streamlit as st # type: ignore
import plotly.graph_objects as go
from sklearn.decomposition import PCA
import pickle

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_core.callbacks import StdOutCallbackHandler
from langchain.schema import Document

# Configuration
MODEL = "gpt-4o-mini"
# Determine the path to the FAISS index relative to this file so the app works
# regardless of the current working directory.
SCRIPT_DIR = Path(__file__).resolve().parent
faiss_db_path = SCRIPT_DIR / "faiss_db"
K_FACTOR = 25

# Load environment variables
load_dotenv()

# Gracefully load the OpenAI API key either from the environment or, if
# available, from Streamlit's secrets file. Accessing ``st.secrets`` when no
# ``secrets.toml`` file exists raises ``StreamlitSecretNotFoundError`` so we
# guard against that scenario.
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    try:
        api_key = st.secrets["OPENAI_API_KEY"]
    except Exception:
        api_key = ""

if not api_key:
    st.warning(
        "OpenAI API key not found. Set OPENAI_API_KEY in environment or Streamlit secrets."
    )
else:
    os.environ["OPENAI_API_KEY"] = api_key

# Page configuration
st.set_page_config(
    page_title="RAG Sustainability Chatbot (FAISS)",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_resource
def initialize_rag_system():
    """Initialize the RAG system using FAISS instead of Chroma"""
    try:
        embeddings = OpenAIEmbeddings(openai_api_key=api_key)
        
        # Try to load existing FAISS database. ``FAISS.save_local`` stores
        # ``index.faiss`` and ``index.pkl`` inside the target directory, so we
        # check for those files instead of ``faiss_db_path.faiss``.
        index_faiss = faiss_db_path / "index.faiss"
        index_pkl = faiss_db_path / "index.pkl"
        if index_faiss.exists() and index_pkl.exists():
            st.info("Loading existing FAISS database...")
            vectorstore = FAISS.load_local(str(faiss_db_path), embeddings, allow_dangerous_deserialization=True)

            # Load metadata
            metadata_path = SCRIPT_DIR / "faiss_db_metadata.pkl"
            with open(metadata_path, 'rb') as f:
                metadata = pickle.load(f)
                doc_texts = metadata['doc_texts']
                doc_types = metadata['doc_types']
                vectors = metadata['vectors']
            
        else:
            st.error("❌ FAISS database not found!")
            st.write("**To create FAISS database from your Chroma database:**")
            st.code("""
# Run this script locally to convert Chroma to FAISS:
python convert_chroma_to_faiss.py
            """)
            return None
        
        # Create PCA for visualization
        pca = PCA(n_components=3)
        reduced_vectors = pca.fit_transform(vectors)
        
        # Initialize LLM and conversation chain
        llm = ChatOpenAI(temperature=0.7, model_name=MODEL, openai_api_key=api_key)
        memory = ConversationBufferMemory(
            memory_key='chat_history', 
            return_messages=True, 
            output_key='answer'
        )
        retriever = vectorstore.as_retriever(search_kwargs={"k": K_FACTOR})
        conversation_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=retriever,
            memory=memory,
            return_source_documents=True,
            callbacks=[StdOutCallbackHandler()],
            output_key="answer",
        )
        
        st.success("✅ FAISS-based RAG system initialized successfully!")
        
        return {
            'embeddings': embeddings,
            'vectorstore': vectorstore,
            'vectors': vectors,
            'doc_texts': doc_texts,
            'doc_types': doc_types,
            'reduced_vectors': reduced_vectors,
            'pca': pca,
            'conversation_chain': conversation_chain
        }
    except Exception as e:
        st.error(f"❌ Error initializing FAISS RAG system: {str(e)}")
        
        # Show detailed error information
        st.error("**Troubleshooting Steps:**")
        st.write("1. Convert your Chroma database to FAISS using the conversion script")
        st.write("2. Ensure FAISS database files are in your repository")
        st.write("3. Check that OpenAI API key is configured")
        
        # Show system information for debugging
        with st.expander("🔍 System Information"):
            st.write(f"Python version: {sys.version}")
            st.write(f"Current directory: {os.getcwd()}")
            st.write(
                "FAISS files exist: "
                f"{(faiss_db_path / 'index.faiss').exists()}"
            )
        
        return None

def create_visualization_plot(reduced_vectors, doc_types, doc_texts, query_vector=None, source_documents=None):
    """Create the 3D visualization plot"""
    unique_doc_types = sorted(set(doc_types))
    hover_texts = [doc[:120].replace('\n', ' ') + "..." for doc in doc_texts]
    
    df = pd.DataFrame({
        'x': reduced_vectors[:, 0],
        'y': reduced_vectors[:, 1],
        'z': reduced_vectors[:, 2],
        'doc_type': doc_types,
        'text': hover_texts,
    })
    
    traces = []
    
    # Add document traces
    if source_documents:
        source_texts = set(doc.page_content for doc in source_documents)
        df['is_retrieved'] = [doc in source_texts for doc in doc_texts]
        
        # Non-retrieved documents (dimmed)
        for doc_type in unique_doc_types:
            group = df[(df['doc_type'] == doc_type) & (~df['is_retrieved'])]
            if not group.empty:
                traces.append(
                    go.Scatter3d(
                        x=group['x'], y=group['y'], z=group['z'],
                        mode='markers', name=doc_type,
                        text=group['text'],
                        hovertemplate="<b>%{text}</b><extra></extra>",
                        marker=dict(size=2, opacity=0.3),
                    )
                )
        
        # Retrieved documents (highlighted)
        retrieved_df = df[df['is_retrieved']]
        if not retrieved_df.empty:
            traces.append(
                go.Scatter3d(
                    x=retrieved_df['x'], y=retrieved_df['y'], z=retrieved_df['z'],
                    mode='markers', name='Retrieved Chunks',
                    text=retrieved_df['text'],
                    hovertemplate="<b>%{text}</b><extra></extra>",
                    marker=dict(size=6, color='white', opacity=1.0),
                )
            )
        
        # Query point
        if query_vector is not None:
            traces.append(
                go.Scatter3d(
                    x=[query_vector[0]], y=[query_vector[1]], z=[query_vector[2]],
                    mode='markers', name='Your Query',
                    text=["Your Query"],
                    hovertemplate="<b>%{text}</b><extra></extra>",
                    marker=dict(size=8, color='yellow', symbol='diamond'),
                )
            )
    else:
        # Initial view - all documents
        for doc_type in unique_doc_types:
            group = df[df['doc_type'] == doc_type]
            traces.append(
                go.Scatter3d(
                    x=group['x'], y=group['y'], z=group['z'],
                    mode='markers', name=doc_type,
                    text=group['text'],
                    hovertemplate="<b>%{text}</b><extra></extra>",
                    marker=dict(size=3, opacity=0.7),
                )
            )
    
    layout = go.Layout(
        title="3D Visualization of RAG Knowledge Base (FAISS)",
        height=500,
        scene=dict(
            xaxis=dict(backgroundcolor='rgb(30,30,30)', color='white'),
            yaxis=dict(backgroundcolor='rgb(30,30,30)', color='white'),
            zaxis=dict(backgroundcolor='rgb(30,30,30)', color='white'),
            bgcolor='rgb(20,20,20)'
        ),
        paper_bgcolor='rgb(20,20,20)',
        plot_bgcolor='rgb(20,20,20)',
        font=dict(color='white'),
        legend=dict(x=0.01, y=0.99, bgcolor='rgba(0,0,0,0)', font=dict(size=10)),
    )
    
    return go.Figure(data=traces, layout=layout)

def main():
    """Main Streamlit application"""
    st.title("🌱 RAG Sustainability Chatbot (FAISS)")
    st.markdown("Ask questions about sustainability practices, certifications, and regulations!")
    st.info("This version uses FAISS instead of Chroma to avoid SQLite compatibility issues.")
    
    # Initialize the RAG system
    rag_system = initialize_rag_system()
    if rag_system is None:
        st.stop()
    
    # Sidebar with information
    with st.sidebar:
        st.header("📊 Database Info")
        st.write(f"**Documents:** {len(rag_system['doc_texts'])}")
        st.write(f"**Chunks:** {len(rag_system['doc_texts'])}")
        st.write(f"**K Factor:** {K_FACTOR}")
        st.write("**Powered by:** FAISS + OpenAI")
        
        st.header("💡 Sample Questions")
        sample_questions = [
            "What are the key differences between UK and US green claims guidelines?",
            "How much water is needed to produce a cotton T-shirt?",
            "What is the EU's Extended Producer Responsibility scheme for textiles?",
            "Compare GOTS and OEKO-TEX certifications for sustainable textiles"
        ]
        
        for i, question in enumerate(sample_questions):
            if st.button(f"Q{i+1}: {question[:50]}...", key=f"sample_{i}"):
                st.session_state.user_question = question
    
    # Main content area
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("💬 Chat")
        
        # Chat interface
        if "messages" not in st.session_state:
            st.session_state.messages = []
        
        # Display chat history
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.write(message["content"])
        
        # Chat input
        if prompt := st.chat_input("Ask a sustainability question..."):
            st.session_state.user_question = prompt
        
        # Handle question from sidebar or chat input
        if hasattr(st.session_state, 'user_question'):
            question = st.session_state.user_question
            
            # Add user message to chat
            st.session_state.messages.append({"role": "user", "content": question})
            with st.chat_message("user"):
                st.write(question)
            
            # Process the question
            with st.chat_message("assistant"):
                with st.spinner("Searching knowledge base..."):
                    try:
                        result = rag_system['conversation_chain'].invoke({"question": question})
                        answer = result["answer"]
                        source_documents = result.get("source_documents", [])
                        
                        st.write(answer)
                        
                        # Store assistant response
                        st.session_state.messages.append({"role": "assistant", "content": answer})
                        
                        # Update visualization
                        query_vector = rag_system['embeddings'].embed_query(question)
                        reduced_query = rag_system['pca'].transform([query_vector])[0]
                        
                        st.session_state.current_plot = create_visualization_plot(
                            rag_system['reduced_vectors'],
                            rag_system['doc_types'],
                            rag_system['doc_texts'],
                            reduced_query,
                            source_documents
                        )
                        st.session_state.source_documents = source_documents
                        
                    except Exception as e:
                        st.error(f"Error processing question: {str(e)}")
            
            # Clear the question state
            del st.session_state.user_question
    
    with col2:
        st.subheader("🎯 Vector Space Visualization")
        
        # Display the plot
        if hasattr(st.session_state, 'current_plot'):
            st.plotly_chart(st.session_state.current_plot, use_container_width=True)
        else:
            # Initial plot
            initial_plot = create_visualization_plot(
                rag_system['reduced_vectors'],
                rag_system['doc_types'],
                rag_system['doc_texts']
            )
            st.plotly_chart(initial_plot, use_container_width=True)
        
        # Display source documents if available
        if hasattr(st.session_state, 'source_documents') and st.session_state.source_documents:
            st.subheader("📦 Retrieved Knowledge Chunks")
            for i, doc in enumerate(st.session_state.source_documents[:5]):  # Show first 5
                with st.expander(f"Chunk {i+1} - {doc.metadata.get('doc_type', 'unknown')}"):
                    st.write(doc.page_content[:500] + "..." if len(doc.page_content) > 500 else doc.page_content)

if __name__ == "__main__":
    main() 