"""
Vector Database Builder for RAG Sustainability Chatbot

This script builds or rebuilds the vector database from the knowledge base.
Use this when:
- Setting up the system for the first time
- Adding new documents to the knowledge base
- Updating chunking parameters
- Recreating the database after corruption

Author: RAG Sustainability Project
"""

import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings

from chunker import create_chunks_for_vectorstore
from vector_store import create_vector_store_from_chunks


def main():
    """
    Main function to build the vector database.
    """
    print("🔨 RAG Sustainability Chatbot - Database Builder")
    print("=" * 50)
    
    try:
        # Load environment variables
        load_dotenv()
        
        # Check for OpenAI API key
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            print("❌ Error: OPENAI_API_KEY not found in environment variables")
            print("💡 Please create a .env file with your OpenAI API key")
            return
        
        print("✅ Environment variables loaded")
        
        # Initialize embeddings
        print("🔗 Initializing OpenAI embeddings...")
        embeddings = OpenAIEmbeddings()
        
        # Process documents into chunks
        print("📚 Processing knowledge base documents...")
        chunks = create_chunks_for_vectorstore(
            chunk_size=1600,
            chunk_overlap=400,
            knowledge_base_path="knowledge-base",
            create_preview=True  # Creates chunk_preview.csv for inspection
        )
        
        if not chunks:
            print("❌ Error: No chunks created from knowledge base")
            print("💡 Check that knowledge-base/ directory contains .txt files")
            return
        
        print(f"✅ Created {len(chunks)} chunks from knowledge base")
        
        # Create vector store
        print("🗄️ Building vector database...")
        vector_store_manager = create_vector_store_from_chunks(
            chunks=chunks,
            db_name="vector_db",
            embeddings=embeddings,
            replace_existing=True  # This will overwrite existing database
        )
        
        # Get database info
        db_info = vector_store_manager.get_database_info()
        
        print("\n🎉 Database built successfully!")
        print(f"📊 Database Statistics:")
        print(f"   • Total chunks: {db_info['total_chunks']}")
        print(f"   • Document types: {db_info['unique_doc_types']}")
        print(f"   • Vector dimensions: {db_info['vector_dimensions']}")
        print(f"   • Database path: {db_info['database_path']}")
        
        print("\n💡 Next steps:")
        print("   1. Run 'python app.py' to launch the chatbot")
        print("   2. Check 'chunk_preview.csv' to inspect document chunks")
        print("   3. Ask questions about sustainability!")
        
    except FileNotFoundError as e:
        print(f"❌ Error: Required files not found - {e}")
        print("💡 Make sure knowledge-base/ directory exists with .txt files")
        
    except Exception as e:
        print(f"❌ Error building database: {e}")
        print("💡 Check your OpenAI API key and internet connection")
        raise


if __name__ == "__main__":
    main() 