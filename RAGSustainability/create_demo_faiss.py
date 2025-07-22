#!/usr/bin/env python3
"""
Create a demo FAISS database for testing Streamlit Cloud deployment
This creates a small working example to test the deployment process
"""

import os
import pickle
import numpy as np
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import Document

# Load environment variables
load_dotenv()

def create_demo_faiss():
    """Create a small demo FAISS database for testing"""
    
    print("🎭 Creating Demo FAISS Database for Deployment Testing")
    print("=" * 60)
    
    try:
        # Sample sustainability documents for testing
        sample_docs = [
            Document(
                page_content="The UK Green Claims Code provides guidance for businesses making environmental claims about their products and services. It requires claims to be truthful, clear, and substantiated with evidence.",
                metadata={"doc_type": "UK_Gov", "source": "Green Claims Code"}
            ),
            Document(
                page_content="The EU's Circular Economy Action Plan aims to accelerate the transition to a regenerative growth model that gives back to the planet more than it takes.",
                metadata={"doc_type": "European_Commission", "source": "Circular Economy"}
            ),
            Document(
                page_content="GOTS (Global Organic Textile Standard) is the worldwide leading textile processing standard for organic fibers, including ecological and social requirements.",
                metadata={"doc_type": "Global_Standard_gGmbH", "source": "GOTS Certification"}
            ),
            Document(
                page_content="Fast fashion contributes significantly to water pollution, with textile dyeing being the second largest polluter of water globally after agriculture.",
                metadata={"doc_type": "Environmental_Justice_Foundation", "source": "Textile Impact"}
            ),
            Document(
                page_content="The production of a single cotton T-shirt requires approximately 2,700 liters of water, equivalent to what one person drinks in 2.5 years.",
                metadata={"doc_type": "Fashion_Revolution", "source": "Water Impact"}
            ),
            Document(
                page_content="OEKO-TEX Standard 100 tests for harmful substances and chemicals in textiles to ensure human-ecological safety.",
                metadata={"doc_type": "OEKO-TEX_Association", "source": "Standard 100"}
            ),
            Document(
                page_content="The FTC Green Guides provide guidance on environmental marketing claims in the United States, helping businesses avoid deceptive advertising.",
                metadata={"doc_type": "US_Federal_Trade_Commission", "source": "Green Guides"}
            ),
            Document(
                page_content="Cradle to Cradle Certified products are designed for the circular economy, considering material health, renewable energy use, and social fairness.",
                metadata={"doc_type": "Cradle_to_Cradle", "source": "C2C Certification"}
            ),
            Document(
                page_content="The Fashion Transparency Index evaluates and ranks fashion brands based on how much they disclose about their supply chain practices.",
                metadata={"doc_type": "Fashion_Revolution", "source": "Transparency Index"}
            ),
            Document(
                page_content="Better Cotton Initiative promotes better standards in cotton farming and practices across 23 countries, making cotton production better for the environment.",
                metadata={"doc_type": "Better_Cotton", "source": "Sustainability Standards"}
            )
        ]
        
        print(f"📝 Created {len(sample_docs)} sample documents")
        
        # Initialize embeddings
        print("🔗 Initializing OpenAI embeddings...")
        embeddings = OpenAIEmbeddings()
        
        # Create FAISS database
        print("🏗️  Building FAISS database...")
        faiss_db = FAISS.from_documents(sample_docs, embeddings)
        
        # Save FAISS database
        faiss_path = "faiss_db"
        print(f"💾 Saving FAISS database to {faiss_path}...")
        faiss_db.save_local(faiss_path)
        
        # Create embeddings for visualization
        print("📊 Creating embeddings for visualization...")
        doc_texts = [doc.page_content for doc in sample_docs]
        doc_types = [doc.metadata['doc_type'] for doc in sample_docs]
        
        # Get embeddings for all documents
        embeddings_list = []
        for doc_text in doc_texts:
            embedding = embeddings.embed_query(doc_text)
            embeddings_list.append(embedding)
        
        embeddings_array = np.array(embeddings_list)
        
        # Save metadata for visualization
        metadata_dict = {
            'doc_texts': doc_texts,
            'doc_types': doc_types,
            'vectors': embeddings_array
        }
        
        with open(f"{faiss_path}_metadata.pkl", 'wb') as f:
            pickle.dump(metadata_dict, f)
        
        print("✅ Demo FAISS database created successfully!")
        print(f"📁 Files created:")
        print(f"   - {faiss_path}.faiss")
        print(f"   - {faiss_path}.pkl") 
        print(f"   - {faiss_path}_metadata.pkl")
        
        # Test the database
        print("🧪 Testing FAISS database...")
        test_queries = [
            "What is the UK Green Claims Code?",
            "How much water does a cotton T-shirt require?",
            "What certifications exist for sustainable textiles?"
        ]
        
        for query in test_queries:
            results = faiss_db.similarity_search(query, k=2)
            print(f"   Query: '{query}' → Found {len(results)} results")
        
        return True
        
    except Exception as e:
        print(f"❌ Error creating demo database: {str(e)}")
        return False

if __name__ == "__main__":
    success = create_demo_faiss()
    
    if success:
        print("\n🎉 Demo database created! Next steps:")
        print("1. Test locally: streamlit run streamlit_app_faiss.py")
        print("2. Commit files to git: git add faiss_db.*")
        print("3. Push to GitHub and deploy on Streamlit Cloud")
        print("4. Use streamlit_app_faiss.py as your main file")
    else:
        print("\n💥 Demo creation failed.")
