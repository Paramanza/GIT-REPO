"""
Document Chunking Module for RAG Sustainability Chatbot

This module handles the complete document processing pipeline:
1. Loading documents from the knowledge base directory structure
2. Chunking documents with appropriate overlap for optimal retrieval
3. Adding metadata based on document source organization
4. Creating debug/preview files for analysis

The chunker supports automatic encoding detection for international documents
and semantic splitting that respects document structure.

Author: RAG Sustainability Project
"""

import os
import glob
import csv
from typing import List, Dict, Tuple
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema import Document


class SmartTextLoader(TextLoader):
    """
    Enhanced TextLoader with automatic encoding detection.
    
    Handles documents with various encodings commonly found in
    sustainability and policy documents from different regions.
    """
    
    def __init__(self, file_path: str, autodetect_encoding: bool = True):
        """
        Initialize the SmartTextLoader.
        
        Args:
            file_path: Path to the text file to load
            autodetect_encoding: Whether to auto-detect encoding (recommended)
        """
        if autodetect_encoding:
            with open(file_path, "rb") as f:
                import chardet
                result = chardet.detect(f.read())
                encoding = result['encoding'] or 'utf-8'
        else:
            encoding = 'utf-8'
        super().__init__(file_path, encoding=encoding)


class DocumentChunker:
    """
    Manages the complete document preparation pipeline for the RAG system.
    
    This class handles:
    - Loading documents from organized knowledge base folders
    - Chunking documents with semantic awareness
    - Adding metadata based on document source organization
    - Creating debug/preview files for analysis
    """
    
    def __init__(
        self, 
        chunk_size: int = 1600,
        chunk_overlap: int = 400,
        knowledge_base_path: str = "knowledge-base"
    ):
        """
        Initialize the DocumentChunker.
        
        Args:
            chunk_size: Size of each text chunk in tokens
            chunk_overlap: Number of tokens to overlap between chunks
            knowledge_base_path: Path to the knowledge base directory
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.knowledge_base_path = knowledge_base_path
        
        # Initialize text splitter with semantic separators
        # Prioritizes paragraph breaks, then sentences, then words
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ".", " "]
        )
    
    def load_documents(self) -> Tuple[List[Document], Dict[str, str]]:
        """
        Load all documents from the knowledge base directory structure.
        
        Each subdirectory represents an organization (e.g., "EU_Parliament",
        "Carbon_Trust") and contains relevant documents as .txt files.
        
        Returns:
            Tuple of (documents_list, folder_mapping_dict)
            - documents_list: List of loaded Document objects
            - folder_mapping_dict: Maps folder paths to document types
        """
        folders = glob.glob(f"{self.knowledge_base_path}/*")
        documents = []
        folder_mapping = {}
        
        print(f"Loading documents from {len(folders)} organizations...")
        
        for folder in folders:
            if not os.path.isdir(folder):
                continue
                
            # Extract organization name from folder
            doc_type = os.path.basename(folder)
            folder_mapping[folder] = doc_type
            
            # Load all text files from this organization's folder
            loader = DirectoryLoader(
                folder,
                glob="**/*.txt",
                loader_cls=SmartTextLoader
            )
            
            try:
                folder_docs = loader.load()
                documents.extend(folder_docs)
                print(f"  ✓ {doc_type}: {len(folder_docs)} documents")
            except Exception as e:
                print(f"  ✗ {doc_type}: Error loading - {e}")
        
        print(f"Total documents loaded: {len(documents)}")
        return documents, folder_mapping
    
    def add_metadata_to_chunks(
        self, 
        chunks: List[Document], 
        folder_mapping: Dict[str, str]
    ) -> List[Document]:
        """
        Add metadata to chunks based on their source document organization.
        
        This enriches each chunk with:
        - doc_type: The organization/source type (e.g., "EU_Parliament", "Carbon_Trust")
        - Original source path information for traceability
        
        Args:
            chunks: List of document chunks to enrich
            folder_mapping: Maps folder paths to document types
            
        Returns:
            List of chunks with enriched metadata
        """
        for chunk in chunks:
            source_path = chunk.metadata.get('source', '')
            
            # Find which organization folder this chunk originated from
            for folder_path, doc_type in folder_mapping.items():
                if folder_path in source_path:
                    chunk.metadata["doc_type"] = doc_type
                    break
            else:
                # Fallback if no match found
                chunk.metadata["doc_type"] = "unknown"
        
        return chunks
    
    def create_chunks(
        self, 
        documents: List[Document], 
        folder_mapping: Dict[str, str]
    ) -> List[Document]:
        """
        Convert documents into chunks with metadata.
        
        Uses semantic-aware splitting that respects document structure,
        then adds organization metadata to each chunk.
        
        Args:
            documents: List of loaded documents
            folder_mapping: Maps folder paths to document types
            
        Returns:
            List of document chunks ready for vectorization
        """
        print("Chunking documents...")
        chunks = self.text_splitter.split_documents(documents)
        chunks = self.add_metadata_to_chunks(chunks, folder_mapping)
        
        print(f"Created {len(chunks)} chunks from {len(documents)} documents")
        return chunks
    
    def create_chunk_preview(self, chunks: List[Document], output_file: str = "chunk_preview.csv"):
        """
        Create a CSV preview of chunks for debugging and analysis.
        
        This is useful for:
        - Understanding how documents are being chunked
        - Debugging retrieval issues
        - Analyzing chunk distribution across organizations
        
        Args:
            chunks: List of document chunks
            output_file: Path for the output CSV file
        """
        print(f"Creating chunk preview: {output_file}")
        
        with open(output_file, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Index", "Doc Type", "Length", "Content"])
            
            for i, chunk in enumerate(chunks):
                writer.writerow([
                    i + 1,
                    chunk.metadata.get("doc_type", "unknown"),
                    len(chunk.page_content),
                    chunk.page_content.replace("\n", " ")
                ])
        
        print(f"Preview saved with {len(chunks)} chunks")
    
    def process_knowledge_base(self, create_preview: bool = False) -> List[Document]:
        """
        Complete document processing pipeline.
        
        This is the main entry point that orchestrates:
        1. Loading documents from the knowledge base
        2. Chunking with semantic awareness
        3. Adding metadata
        4. Optionally creating a preview file
        
        Args:
            create_preview: Whether to create a CSV preview file
            
        Returns:
            List of processed chunks ready for vector storage
            
        Raises:
            ValueError: If no documents found in knowledge base
        """
        # Load all documents
        documents, folder_mapping = self.load_documents()
        
        if not documents:
            raise ValueError("No documents found in knowledge base")
        
        # Create chunks with metadata
        chunks = self.create_chunks(documents, folder_mapping)
        
        # Optionally create preview file
        if create_preview:
            self.create_chunk_preview(chunks)
        
        return chunks


def create_chunks_for_vectorstore(
    chunk_size: int = 1600,
    chunk_overlap: int = 400,
    knowledge_base_path: str = "knowledge-base",
    create_preview: bool = False
) -> List[Document]:
    """
    Convenience function to create chunks ready for vector storage.
    
    This provides a simple interface for the most common use case:
    processing all documents in the knowledge base with standard parameters.
    
    Args:
        chunk_size: Size of each text chunk in tokens
        chunk_overlap: Number of tokens to overlap between chunks  
        knowledge_base_path: Path to the knowledge base directory
        create_preview: Whether to create a CSV preview file
        
    Returns:
        List of processed chunks ready for vector storage
    """
    chunker = DocumentChunker(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        knowledge_base_path=knowledge_base_path
    )
    
    return chunker.process_knowledge_base(create_preview=create_preview)


if __name__ == "__main__":
    """
    Standalone script to process documents and create chunks.
    
    Run this script directly to process the knowledge base and create
    a preview file for inspection.
    """
    print("RAG Document Chunker")
    print("===================")
    
    try:
        chunks = create_chunks_for_vectorstore(create_preview=True)
        print(f"\n✅ Successfully processed {len(chunks)} chunks")
        print("Chunks are ready for vector storage")
        print("📋 Check chunk_preview.csv to inspect the results")
    except Exception as e:
        print(f"\n❌ Error processing documents: {e}")
        print("💡 Make sure the knowledge-base directory exists with .txt files") 