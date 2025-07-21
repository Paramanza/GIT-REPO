#!/usr/bin/env python3
"""
Convert Chroma vector database to FAISS format
This avoids SQLite dependency issues on Streamlit Cloud
"""

import os
import sys
import numpy as np
import pickle
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_community.vectorstores import FAISS
from langchain.schema import Document

# Load environment variables
load_dotenv()

def convert_chroma_to_faiss():
    """Convert existing Chroma database to FAISS format"""
    
    print("🔄 Converting Chroma database to FAISS...")
    
    try:
        # Initialize embeddings
        embeddings = OpenAIEmbeddings()
        
        # Load existing Chroma database
        chroma_db = "vector_db"
        if not os.path.exists(chroma_db):
            print(f"❌ Chroma database not found at {chroma_db}")
            return False
        
        print("📂 Loading Chroma database...")
        vectorstore = Chroma(persist_directory=chroma_db, embedding_function=embeddings)
        
        # Get all documents and their embeddings
        collection = vectorstore._collection
        result = collection.get(include=['embeddings', 'documents', 'metadatas'])
        
        documents = result['documents']
        metadatas = result['metadatas']
        embeddings_array = np.array(result['embeddings'])
        
        print(f"📊 Found {len(documents)} documents")
        
        # Create LangChain Document objects
        langchain_docs = []
        for i, (doc_text, metadata) in enumerate(zip(documents, metadatas)):
            langchain_docs.append(Document(page_content=doc_text, metadata=metadata))
        
        # Create FAISS database from documents
        print("🏗️  Building FAISS database...")
        faiss_db = FAISS.from_documents(langchain_docs, embeddings)
        
        # Save FAISS database
        faiss_path = "faiss_db"
        print(f"💾 Saving FAISS database to {faiss_path}...")
        faiss_db.save_local(faiss_path)
        
        # Save additional metadata for visualization
        doc_types = [metadata.get('doc_type', 'unknown') for metadata in metadatas]
        metadata_dict = {
            'doc_texts': documents,
            'doc_types': doc_types,
            'vectors': embeddings_array
        }
        
        with open(f"{faiss_path}_metadata.pkl", 'wb') as f:
            pickle.dump(metadata_dict, f)
        
        print("✅ Conversion completed successfully!")
        print(f"📁 FAISS files created:")
        print(f"   - {faiss_path}.faiss")
        print(f"   - {faiss_path}.pkl")
        print(f"   - {faiss_path}_metadata.pkl")
        
        # Test the FAISS database
        print("🧪 Testing FAISS database...")
        test_query = "sustainability practices"
        results = faiss_db.similarity_search(test_query, k=3)
        print(f"   Found {len(results)} results for test query")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during conversion: {str(e)}")
        return False

if __name__ == "__main__":
    print("🌱 Chroma to FAISS Converter")
    print("=" * 40)
    
    success = convert_chroma_to_faiss()
    
    if success:
        print("\n🎉 Conversion completed! You can now:")
        print("1. Add the FAISS files to your git repository")
        print("2. Use streamlit_app_faiss.py for deployment")
        print("3. Deploy to Streamlit Cloud without SQLite issues")
    else:
        print("\n💥 Conversion failed. Please check the error messages above.")
        sys.exit(1) 