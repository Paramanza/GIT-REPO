"""
RAG Conversation Chain Module for RAG Sustainability Chatbot

This module handles the core RAG (Retrieval Augmented Generation) functionality:

1. Conversation Management:
   - LangChain conversation chains with memory
   - Query processing and response generation
   - Conversation history tracking

2. Response Processing:
   - Formatting responses for UI display
   - Chunk display and metadata extraction
   - Error handling and fallback responses

3. Sample Questions:
   - Curated test questions for system evaluation
   - Categorized by query complexity and type

The module uses OpenAI's GPT models for generation and integrates with
the vector store for relevant context retrieval.

Author: RAG Sustainability Project
"""

from typing import List, Dict, Any, Tuple, Optional
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.schema import Document
from langchain_core.callbacks import StdOutCallbackHandler


class RAGConversationManager:
    """
    Manages RAG conversations with memory and response formatting.
    
    This class orchestrates the entire conversation flow:
    - Setting up LangChain conversation chains
    - Managing conversation memory across interactions
    - Processing queries and retrieving relevant context
    - Formatting responses for display in the UI
    """
    
    def __init__(
        self,
        retriever,
        model_name: str = "gpt-4o-mini",
        temperature: float = 0.7,
        k_factor: int = 25,
        enable_callbacks: bool = True
    ):
        """
        Initialize the RAG conversation manager.
        
        Args:
            retriever: LangChain retriever instance from vector store
            model_name: OpenAI model to use for generation
            temperature: Creativity level (0-1, higher = more creative)
            k_factor: Number of chunks to retrieve per query
            enable_callbacks: Whether to enable verbose output callbacks
        """
        self.model_name = model_name
        self.temperature = temperature
        self.k_factor = k_factor
        
        # Initialize language model
        self.llm = ChatOpenAI(
            temperature=temperature,
            model_name=model_name
        )
        
        # Setup conversation memory
        self.memory = ConversationBufferMemory(
            memory_key='chat_history',
            return_messages=True,
            output_key='answer'
        )
        
        # Setup callbacks for debugging
        callbacks = [StdOutCallbackHandler()] if enable_callbacks else []
        
        # Create conversation chain
        self.conversation_chain = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=retriever,
            memory=self.memory,
            return_source_documents=True,
            callbacks=callbacks,
            output_key="answer",
        )
    
    def query(self, question: str) -> Dict[str, Any]:
        """
        Process a user query using the RAG chain.
        
        Args:
            question: User's question/query
            
        Returns:
            Dictionary containing:
            - answer: Generated response
            - source_documents: Retrieved document chunks
            - question: Original question
        """
        if not question.strip():
            return {
                "answer": "Please provide a question.",
                "source_documents": [],
                "question": question
            }
        
        try:
            # Get response from conversation chain
            result = self.conversation_chain.invoke({"question": question})
            
            return {
                "answer": result["answer"],
                "source_documents": result.get("source_documents", []),
                "question": question
            }
        
        except Exception as e:
            return {
                "answer": f"Sorry, I encountered an error: {str(e)}",
                "source_documents": [],
                "question": question
            }
    
    def format_chunks_for_display(
        self, 
        source_documents: List[Document],
        max_chunk_length: int = 500
    ) -> str:
        """
        Format retrieved chunks for display in the UI.
        
        Args:
            source_documents: Retrieved document chunks
            max_chunk_length: Maximum characters to display per chunk
            
        Returns:
            Formatted string showing all chunks with metadata
        """
        if not source_documents:
            return "No relevant chunks retrieved."
        
        chunk_texts = []
        for i, doc in enumerate(source_documents):
            # Extract metadata
            doc_type = doc.metadata.get('doc_type', 'unknown')
            
            # Create chunk header
            chunk_text = f"**Chunk {i+1}** (from {doc_type}):\n"
            
            # Add truncated content
            content = doc.page_content
            if len(content) > max_chunk_length:
                content = content[:max_chunk_length] + "..."
            
            chunk_text += content
            chunk_texts.append(chunk_text)
        
        # Join all chunks with separators
        separator = "\n\n" + "="*50 + "\n\n"
        return separator.join(chunk_texts)
    
    def get_conversation_history(self) -> List[Dict[str, str]]:
        """
        Get the current conversation history.
        
        Returns:
            List of conversation turns with 'user' and 'assistant' messages
        """
        history = []
        messages = self.memory.chat_memory.messages
        
        for i in range(0, len(messages), 2):
            if i + 1 < len(messages):
                user_msg = messages[i].content
                assistant_msg = messages[i + 1].content
                history.append({
                    "user": user_msg,
                    "assistant": assistant_msg
                })
        
        return history
    
    def clear_memory(self):
        """Clear the conversation memory."""
        self.memory.clear()
    
    def get_memory_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the current memory state.
        
        Returns:
            Dictionary with memory statistics
        """
        messages = self.memory.chat_memory.messages
        return {
            "total_messages": len(messages),
            "conversation_turns": len(messages) // 2,
            "memory_key": self.memory.memory_key,
            "has_history": len(messages) > 0
        }


class RAGResponseProcessor:
    """
    Processes and formats RAG responses for different UI contexts.
    
    This class provides static methods for common response processing
    tasks that can be used across different interface types.
    """
    
    @staticmethod
    def process_chat_response(
        rag_result: Dict[str, Any],
        chat_history: List[List[str]],
        max_chunk_length: int = 500
    ) -> Tuple[List[List[str]], str]:
        """
        Process a RAG result for chat interface display.
        
        Args:
            rag_result: Result from RAG conversation manager
            chat_history: Current chat history (list of [user, assistant] pairs)
            max_chunk_length: Maximum characters per chunk in display
            
        Returns:
            Tuple of (updated_chat_history, formatted_chunks)
        """
        question = rag_result["question"]
        answer = rag_result["answer"]
        source_documents = rag_result["source_documents"]
        
        # Update chat history
        updated_history = chat_history + [[question, answer]]
        
        # Format chunks for display
        if source_documents:
            chunk_texts = []
            for i, doc in enumerate(source_documents):
                doc_type = doc.metadata.get('doc_type', 'unknown')
                
                chunk_text = f"**Chunk {i+1}** (from {doc_type}):\n"
                content = doc.page_content
                if len(content) > max_chunk_length:
                    content = content[:max_chunk_length] + "..."
                chunk_text += content
                chunk_texts.append(chunk_text)
            
            formatted_chunks = "\n\n" + "="*50 + "\n\n".join(chunk_texts)
        else:
            formatted_chunks = "No relevant chunks retrieved."
        
        return updated_history, formatted_chunks
    
    @staticmethod
    def extract_chunk_metadata(source_documents: List[Document]) -> Dict[str, Any]:
        """
        Extract useful metadata from retrieved chunks.
        
        Args:
            source_documents: Retrieved document chunks
            
        Returns:
            Dictionary with aggregated metadata statistics
        """
        if not source_documents:
            return {"total_chunks": 0, "doc_types": [], "sources": []}
        
        doc_types = []
        sources = []
        total_length = 0
        
        for doc in source_documents:
            doc_type = doc.metadata.get('doc_type', 'unknown')
            source = doc.metadata.get('source', 'unknown')
            
            doc_types.append(doc_type)
            sources.append(source)
            total_length += len(doc.page_content)
        
        return {
            "total_chunks": len(source_documents),
            "doc_types": list(set(doc_types)),
            "unique_doc_types": len(set(doc_types)),
            "sources": sources,
            "unique_sources": len(set(sources)),
            "total_content_length": total_length,
            "avg_chunk_length": total_length / len(source_documents)
        }


def create_rag_conversation_manager(
    retriever,
    model_name: str = "gpt-4o-mini",
    temperature: float = 0.7,
    enable_callbacks: bool = True
) -> RAGConversationManager:
    """
    Convenience function to create a RAG conversation manager.
    
    Args:
        retriever: LangChain retriever instance from vector store
        model_name: OpenAI model to use for generation
        temperature: Creativity level (0-1, higher = more creative)
        enable_callbacks: Whether to enable verbose output callbacks
        
    Returns:
        Configured RAGConversationManager instance
    """
    return RAGConversationManager(
        retriever=retriever,
        model_name=model_name,
        temperature=temperature,
        enable_callbacks=enable_callbacks
    )


# ============================================================================
# SAMPLE TEST QUESTIONS
# ============================================================================

SAMPLE_TEST_QUESTIONS = [
    # Cross-Document Comparison Questions
    "What are the key differences between the UK's Green Claims Code and the US FTC Green Guides regarding the use of terms like 'eco-friendly' in marketing fashion products?",
    
    "Compare how the EU and the UK approach the responsibility of fashion businesses in substantiating environmental claims. What legal or practical requirements are in place in both regions?",
    
    "According to EU and UK guidance, what role should consumers play in identifying misleading environmental claims, and how are they supported in this role?",
    
    # Sustainability Metrics and Impact Analysis
    "How much water is required to produce a cotton T-shirt, and how does this illustrate the environmental impact of textile production in the EU?",
    
    "What proportion of clothes are recycled into new garments in the EU, and how does this relate to the EU's goals for a circular economy by 2050?",
    
    "What specific environmental harms are associated with synthetic textiles in both EU and US contexts, and how do policies in each region attempt to mitigate them?",
    
    # Policy and Regulatory Questions  
    "What is the EU's Extended Producer Responsibility (EPR) scheme for textiles, and how does its proposed timeline compare with UK or US regulatory efforts?",
    
    "What are the legal consequences for fashion businesses in the UK that make misleading green claims, and how do these compare to enforcement under the US FTC guidelines?",
    
    # Application & Practical Evaluation
    "If a fashion brand claims that a polyester jacket is 'sustainable' because it uses recycled materials, what further information must they provide according to UK and EU guidance to avoid misleading consumers?",
    
    "What should a fashion brand include on its product labels and marketing to comply with both the EU's proposed Digital Product Passport and the FTC's Green Guides?"
]


def get_sample_questions() -> List[str]:
    """
    Get a list of sample questions for testing the RAG system.
    
    These questions are designed to test different aspects of the RAG system:
    - Cross-document comparison and synthesis
    - Factual recall and metric analysis
    - Policy interpretation and regulatory comparison
    - Practical application of guidelines
    
    Returns:
        List of sample questions covering different query types
    """
    return SAMPLE_TEST_QUESTIONS.copy() 