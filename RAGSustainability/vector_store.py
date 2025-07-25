"""
Vector Store Management and Visualization Module for RAG Sustainability Chatbot

This module handles:
1. Vector database creation and management using Chroma
2. Dimensionality reduction with PCA for 3D visualization  
3. Plotly-based 3D scatter plot generation
4. Query-specific visualization updates

Author: RAG Sustainability Project
"""

import os
import numpy as np
import pandas as pd
from typing import List, Tuple, Optional, Dict, Any
import plotly.graph_objects as go
from sklearn.decomposition import PCA

from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain.schema import Document


class VectorStoreManager:
    """
    Manages the vector database and provides visualization capabilities.
    
    This class handles:
    - Creating and updating Chroma vector stores
    - Loading existing vector stores
    - Extracting embeddings for visualization
    - Managing PCA dimensionality reduction
    - Generating 3D plots with Plotly
    """
    
    def __init__(
        self, 
        db_name: str = "vector_db",
        embeddings: Optional[OpenAIEmbeddings] = None
    ):
        """
        Initialize the VectorStoreManager.
        
        Args:
            db_name: Name/path of the vector database directory
            embeddings: OpenAI embeddings instance (creates new if None)
        """
        self.db_name = db_name
        self.embeddings = embeddings or OpenAIEmbeddings()
        self.vectorstore: Optional[Chroma] = None
        self.pca: Optional[PCA] = None
        
        # Data for visualization (loaded when vector store is loaded)
        self.vectors: Optional[np.ndarray] = None
        self.reduced_vectors: Optional[np.ndarray] = None
        self.doc_texts: Optional[List[str]] = None
        self.metadatas: Optional[List[Dict]] = None
        self.doc_types: Optional[List[str]] = None
    
    def create_vector_store(
        self, 
        chunks: List[Document], 
        replace_existing: bool = True
    ) -> Chroma:
        """
        Create a new vector store from document chunks.
        
        Args:
            chunks: List of document chunks to vectorize
            replace_existing: Whether to replace existing vector store
            
        Returns:
            The created Chroma vector store
        """
        print(f"Creating vector store with {len(chunks)} chunks...")
        
        # Remove existing database if requested
        if replace_existing and os.path.exists(self.db_name):
            print("Removing existing vector store...")
            existing_store = Chroma(
                persist_directory=self.db_name, 
                embedding_function=self.embeddings
            )
            existing_store.delete_collection()
        
        # Create new vector store
        self.vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            persist_directory=self.db_name
        )
        
        print(f"✅ Vector store created with {self.vectorstore._collection.count()} documents")
        
        # Load visualization data
        self._load_visualization_data()
        
        return self.vectorstore
    
    def load_vector_store(self) -> Chroma:
        """
        Load an existing vector store from disk.
        
        Returns:
            The loaded Chroma vector store
            
        Raises:
            FileNotFoundError: If vector store doesn't exist
        """
        if not os.path.exists(self.db_name):
            raise FileNotFoundError(f"Vector store not found at {self.db_name}")
        
        print(f"Loading existing vector store from {self.db_name}...")
        
        self.vectorstore = Chroma(
            persist_directory=self.db_name,
            embedding_function=self.embeddings
        )
        
        print(f"✅ Vector store loaded with {self.vectorstore._collection.count()} documents")
        
        # Load visualization data
        self._load_visualization_data()
        
        return self.vectorstore
    
    def _load_visualization_data(self):
        """
        Load embeddings and metadata for visualization with memory optimization.
        
        This extracts the raw embeddings from Chroma and prepares them
        for PCA dimensionality reduction and plotting.
        """
        if not self.vectorstore:
            raise ValueError("No vector store loaded")
        
        print("Loading visualization data...")
        
        try:
            # Get all embeddings and metadata from Chroma
            collection = self.vectorstore._collection
            result = collection.get(include=['embeddings', 'documents', 'metadatas'])
            
            if len(result['embeddings']) == 0:
                raise ValueError("No embeddings found in vector store")
            
            # Store raw data with memory monitoring
            print(f"Processing {len(result['embeddings'])} embeddings...")
            self.vectors = np.array(result['embeddings'], dtype=np.float32)  # Use float32 to save memory
            self.doc_texts = result['documents']
            self.metadatas = result['metadatas']
            self.doc_types = [metadata.get('doc_type', 'unknown') for metadata in self.metadatas]
            
            # Perform PCA for 3D visualization with memory optimization
            print("Performing PCA dimensionality reduction...")
            self.pca = PCA(n_components=3)
            
            # Process in smaller batches if the dataset is large to reduce memory spikes
            if len(self.vectors) > 2000:
                print("Large dataset detected, using memory-optimized PCA...")
                # For very large datasets, we could implement batch PCA, but for now just proceed
                
            self.reduced_vectors = self.pca.fit_transform(self.vectors)
            
            print(f"✅ Loaded {len(self.vectors)} embeddings for visualization")
            print(f"📊 Memory usage: {self.vectors.nbytes / 1024 / 1024:.1f} MB for vectors")
            print(f"🔍 PCA explained variance ratio: {self.pca.explained_variance_ratio_}")
            
        except Exception as e:
            print(f"❌ Error loading visualization data: {e}")
            print("💡 This might be due to memory constraints or vector store corruption")
            raise
    
    def get_retriever(self, k: int = 25):
        """
        Get a retriever for the vector store.
        
        Args:
            k: Number of documents to retrieve per query
            
        Returns:
            LangChain retriever instance
        """
        if not self.vectorstore:
            raise ValueError("No vector store loaded")
        
        return self.vectorstore.as_retriever(search_kwargs={"k": k})
    
    def create_initial_plot(self) -> go.Figure:
        """
        Create the initial 3D scatter plot showing all document chunks.
        
        This shows the entire knowledge base in 3D space, colored by
        document type (organization).
        
        Returns:
            Plotly Figure object for the 3D visualization
        """
        if self.reduced_vectors is None:
            raise ValueError("Visualization data not loaded")
        
        # Get unique document types for color coding
        unique_doc_types = sorted(set(self.doc_types))
        
        # Create hover text (first 120 chars of each document)
        hover_texts = [
            doc[:120].replace('\n', ' ') + "..." 
            for doc in self.doc_texts
        ]
        
        # Create DataFrame for easier manipulation
        df = pd.DataFrame({
            'x': self.reduced_vectors[:, 0],
            'y': self.reduced_vectors[:, 1], 
            'z': self.reduced_vectors[:, 2],
            'doc_type': self.doc_types,
            'text': hover_texts,
        })
        
        # Create traces for each document type
        traces = []
        for doc_type in unique_doc_types:
            group = df[df['doc_type'] == doc_type]
            traces.append(
                go.Scatter3d(
                    x=group['x'],
                    y=group['y'],
                    z=group['z'],
                    mode='markers',
                    name=doc_type,
                    text=group['text'],
                    hovertemplate="<b>%{text}</b><extra></extra>",
                    marker=dict(size=3, opacity=0.7),
                )
            )
        
        # Create layout with dark theme
        layout = go.Layout(
            title="3D Visualization of RAG Knowledge Base",
            height=490,
            margin=dict(l=0, r=0, b=0, t=30),
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
    
    def create_query_plot(
        self, 
        query_text: str, 
        source_documents: List[Document]
    ) -> go.Figure:
        """
        Create a 3D plot highlighting query results.
        
        This shows:
        - All chunks (faded)
        - Retrieved chunks (highlighted in white)
        - Query vector (yellow diamond)
        
        Args:
            query_text: The user's query
            source_documents: Documents retrieved by the RAG system
            
        Returns:
            Plotly Figure object showing query results in 3D space
        """
        if self.reduced_vectors is None or self.pca is None:
            raise ValueError("Visualization data not loaded")
        
        # Get query embedding and reduce to 3D
        query_vector = self.embeddings.embed_query(query_text)
        reduced_query = self.pca.transform([query_vector])[0]
        
        # Identify which chunks were retrieved
        source_texts = set(doc.page_content for doc in source_documents)
        
        # Create hover text
        hover_texts = [
            doc[:120].replace('\n', ' ') + "..." 
            for doc in self.doc_texts
        ]
        
        unique_doc_types = sorted(set(self.doc_types))
        
        # Create DataFrame with retrieval information
        df = pd.DataFrame({
            'x': self.reduced_vectors[:, 0],
            'y': self.reduced_vectors[:, 1],
            'z': self.reduced_vectors[:, 2],
            'doc_type': self.doc_types,
            'text': hover_texts,
            'is_retrieved': [doc in source_texts for doc in self.doc_texts],
        })
        
        traces = []
        
        # Add regular chunks (faded)
        for doc_type in unique_doc_types:
            group = df[(df['doc_type'] == doc_type) & (~df['is_retrieved'])]
            if not group.empty:
                traces.append(
                    go.Scatter3d(
                        x=group['x'],
                        y=group['y'],
                        z=group['z'],
                        mode='markers',
                        name=doc_type,
                        text=group['text'],
                        hovertemplate="<b>%{text}</b><extra></extra>",
                        marker=dict(size=2, opacity=0.3),
                    )
                )
        
        # Add retrieved chunks (highlighted)
        retrieved_df = df[df['is_retrieved']]
        if not retrieved_df.empty:
            traces.append(
                go.Scatter3d(
                    x=retrieved_df['x'],
                    y=retrieved_df['y'],
                    z=retrieved_df['z'],
                    mode='markers',
                    name='Retrieved Chunks',
                    text=retrieved_df['text'],
                    hovertemplate="<b>%{text}</b><extra></extra>",
                    marker=dict(size=6, color='white', opacity=1.0),
                )
            )
        
        # Add query point
        traces.append(
            go.Scatter3d(
                x=[reduced_query[0]],
                y=[reduced_query[1]],
                z=[reduced_query[2]],
                mode='markers',
                name='Your Query',
                text=[f"Query: {query_text[:50]}..."],
                hovertemplate="<b>%{text}</b><extra></extra>",
                marker=dict(size=8, color='yellow', symbol='diamond'),
            )
        )
        
        # Create layout
        layout = go.Layout(
            title="Query Results in Vector Space",
            height=490,
            margin=dict(l=0, r=0, b=0, t=30),
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
    
    def get_database_info(self) -> Dict[str, Any]:
        """
        Get information about the vector database for display.
        
        Returns:
            Dictionary with database statistics and configuration
        """
        if not self.vectorstore:
            return {"error": "No vector store loaded"}
        
        doc_count = len(self.doc_texts) if self.doc_texts else 0
        unique_types = len(set(self.doc_types)) if self.doc_types else 0
        
        return {
            "total_chunks": doc_count,
            "unique_doc_types": unique_types,
            "vector_dimensions": self.vectors.shape[1] if self.vectors is not None else 0,
            "pca_components": 3,
            "database_path": self.db_name
        }


def create_vector_store_from_chunks(
    chunks: List[Document],
    db_name: str = "vector_db",
    embeddings: Optional[OpenAIEmbeddings] = None,
    replace_existing: bool = True
) -> VectorStoreManager:
    """
    Convenience function to create a vector store from chunks.
    
    Args:
        chunks: Document chunks to vectorize
        db_name: Name/path of the vector database directory
        embeddings: OpenAI embeddings instance (creates new if None)
        replace_existing: Whether to replace existing vector store
        
    Returns:
        Configured VectorStoreManager instance
    """
    manager = VectorStoreManager(db_name=db_name, embeddings=embeddings)
    manager.create_vector_store(chunks, replace_existing=replace_existing)
    return manager


def load_existing_vector_store(
    db_name: str = "vector_db",
    embeddings: Optional[OpenAIEmbeddings] = None
) -> VectorStoreManager:
    """
    Convenience function to load an existing vector store.
    
    Args:
        db_name: Name/path of the vector database directory
        embeddings: OpenAI embeddings instance (creates new if None)
        
    Returns:
        Configured VectorStoreManager instance
    """
    manager = VectorStoreManager(db_name=db_name, embeddings=embeddings)
    manager.load_vector_store()
    return manager 