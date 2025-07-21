#!/usr/bin/env python3
"""
Convert Chroma vector database to FAISS format using direct SQLite access
This avoids SQLite dependency issues by not importing langchain_chroma
"""

import os
import sys
import sqlite3
import numpy as np
import pickle
import json
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import Document

# Load environment variables
load_dotenv()

def read_chroma_database_direct():
    """Read Chroma database directly from SQLite without langchain_chroma"""
    
    print("🔄 Reading Chroma database directly from SQLite...")
    
    # Path to the SQLite database
    db_path = os.path.join("vector_db", "chroma.sqlite3")
    
    if not os.path.exists(db_path):
        print(f"❌ SQLite database not found at {db_path}")
        return None
    
    try:
        # Connect to SQLite database
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Get collection information
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()
        print(f"📊 Found tables: {[table[0] for table in tables]}")
        
        # Try to find the embeddings table
        cursor.execute("SELECT * FROM embeddings LIMIT 1;")
        columns = [description[0] for description in cursor.description]
        print(f"📋 Embeddings table columns: {columns}")
        
        # Get all embeddings data
        cursor.execute("SELECT * FROM embeddings;")
        rows = cursor.fetchall()
        
        documents = []
        metadatas = []
        embeddings_list = []
        
        for row in rows:
            # Parse the row data - adjust indices based on actual schema
            doc_id = row[0]
            embedding_data = row[1] if len(row) > 1 else None
            document_text = row[2] if len(row) > 2 else ""
            metadata_json = row[3] if len(row) > 3 else "{}"
            
            # Parse embedding (might be stored as blob or JSON)
            if embedding_data:
                try:
                    # Try parsing as JSON first
                    if isinstance(embedding_data, str):
                        embedding = json.loads(embedding_data)
                    else:
                        # If it's a blob, convert to numpy array
                        embedding = np.frombuffer(embedding_data, dtype=np.float32).tolist()
                    
                    embeddings_list.append(embedding)
                    documents.append(document_text)
                    
                    # Parse metadata
                    try:
                        metadata = json.loads(metadata_json) if metadata_json else {}
                    except:
                        metadata = {"doc_type": "unknown"}
                    metadatas.append(metadata)
                    
                except Exception as e:
                    print(f"⚠️  Error parsing row {doc_id}: {e}")
                    continue
        
        conn.close()
        
        print(f"✅ Successfully read {len(documents)} documents from Chroma database")
        
        return {
            'documents': documents,
            'metadatas': metadatas,
            'embeddings': embeddings_list
        }
        
    except Exception as e:
        print(f"❌ Error reading SQLite database: {e}")
        return None

def convert_to_faiss():
    """Convert the Chroma data to FAISS format"""
    
    print("🌱 Chroma to FAISS Converter (Direct SQLite)")
    print("=" * 50)
    
    # Read data from Chroma database
    chroma_data = read_chroma_database_direct()
    if not chroma_data:
        return False
    
    try:
        # Initialize embeddings
        embeddings = OpenAIEmbeddings()
        
        # Create LangChain Document objects
        langchain_docs = []
        for doc_text, metadata in zip(chroma_data['documents'], chroma_data['metadatas']):
            langchain_docs.append(Document(page_content=doc_text, metadata=metadata))
        
        print("🏗️  Building FAISS database...")
        # Create FAISS database from documents
        faiss_db = FAISS.from_documents(langchain_docs, embeddings)
        
        # Save FAISS database
        faiss_path = "faiss_db"
        print(f"💾 Saving FAISS database to {faiss_path}...")
        faiss_db.save_local(faiss_path)
        
        # Save additional metadata for visualization
        doc_types = [metadata.get('doc_type', 'unknown') for metadata in chroma_data['metadatas']]
        embeddings_array = np.array(chroma_data['embeddings'])
        
        metadata_dict = {
            'doc_texts': chroma_data['documents'],
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
    success = convert_to_faiss()
    
    if success:
        print("\n🎉 Conversion completed! You can now:")
        print("1. Add the FAISS files to your git repository")
        print("2. Use streamlit_app_faiss.py for deployment")
        print("3. Deploy to Streamlit Cloud without SQLite issues")
    else:
        print("\n💥 Conversion failed. Please check the error messages above.")
        sys.exit(1) 