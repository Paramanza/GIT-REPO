"""
RAG Sustainability Chatbot - Main Gradio Application

This is the main orchestrator for the RAG (Retrieval Augmented Generation) sustainability
chatbot. It provides a Gradio-based web interface with:

1. Interactive chat with memory and context awareness
2. 3D visualization of the vector space using Plotly
3. Real-time highlighting of retrieved document chunks
4. Sample questions for testing the system

The application uses a modular architecture:
- chunker.py: Document loading and text chunking
- vector_store.py: Vector database management and visualization
- rag_chain.py: LangChain conversation logic and response formatting

Author: RAG Sustainability Project
"""

import os
from dotenv import load_dotenv
import gradio as gr

# Import our modular components
from vector_store import load_existing_vector_store
from rag_chain import create_rag_conversation_manager, RAGResponseProcessor, get_sample_questions
from langchain_openai import OpenAIEmbeddings


# ============================================================================
# CONFIGURATION
# ============================================================================

# Model and database configuration
MODEL = "gpt-4o-mini"
DB_NAME = "vector_db"
K_FACTOR = 25  # Number of chunks to retrieve per query
TEMPERATURE = 0.7  # LLM creativity level


# ============================================================================
# INITIALIZATION
# ============================================================================

def initialize_application():
    """
    Initialize the RAG application components.
    
    This function:
    1. Loads environment variables
    2. Sets up OpenAI API key
    3. Loads the existing vector store
    4. Creates the RAG conversation manager
    5. Generates the initial 3D visualization
    
    Returns:
        Tuple of (vector_store_manager, rag_manager, initial_plot)
    """
    print("🚀 Initializing RAG Sustainability Chatbot...")
    
    # Load environment variables
    load_dotenv()
    os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY', 'your-key-if-not-using-env')
    
    # Initialize embeddings
    embeddings = OpenAIEmbeddings()
    
    # Load existing vector store
    print("📚 Loading vector store...")
    vector_store_manager = load_existing_vector_store(
        db_name=DB_NAME,
        embeddings=embeddings
    )
    
    # Create retriever
    retriever = vector_store_manager.get_retriever(k=K_FACTOR)
    
    # Initialize RAG conversation manager
    print("🤖 Setting up conversation chain...")
    rag_manager = create_rag_conversation_manager(
        retriever=retriever,
        model_name=MODEL,
        temperature=TEMPERATURE,
        enable_callbacks=True
    )
    
    # Create initial visualization
    print("📊 Creating initial visualization...")
    initial_plot = vector_store_manager.create_initial_plot()
    
    print("✅ Application initialized successfully!")
    return vector_store_manager, rag_manager, initial_plot


# ============================================================================
# UI INTERACTION HANDLERS
# ============================================================================

def chat_with_rag(message, history, plot_fig, vector_store_manager, rag_manager):
    """
    Handle chat interactions with the RAG system.
    
    This function processes user queries and updates the UI with:
    - Generated responses
    - Retrieved document chunks
    - Updated 3D visualization highlighting relevant chunks
    
    Args:
        message: User's input message
        history: Current chat history
        plot_fig: Current plot figure
        vector_store_manager: Vector store manager instance
        rag_manager: RAG conversation manager instance
        
    Returns:
        Tuple of (updated_history, formatted_chunks, updated_plot)
    """
    # Handle empty messages
    if not message.strip():
        return history, "", plot_fig
    
    # Process query with RAG system
    rag_result = rag_manager.query(message)
    
    # Update chat history and format chunks
    updated_history, formatted_chunks = RAGResponseProcessor.process_chat_response(
        rag_result=rag_result,
        chat_history=history,
        max_chunk_length=500
    )
    
    # Create updated visualization
    updated_plot = vector_store_manager.create_query_plot(
        query_text=message,
        source_documents=rag_result["source_documents"]
    )
    
    return updated_history, formatted_chunks, updated_plot


def create_database_info_text(vector_store_manager):
    """
    Create formatted text showing database information.
    
    Args:
        vector_store_manager: Vector store manager instance
        
    Returns:
        Formatted string with database statistics
    """
    info = vector_store_manager.get_database_info()
    
    return f"""**RAG Vector Database Info:**
• Powered by Chroma                                    • Total Chunks: {info.get('total_chunks', 0)}
• Chunk Size: 1600 tokens                            • Document Types: {info.get('unique_doc_types', 0)}
• Chunk Overlap: 400 tokens                       • K Factor: {K_FACTOR} retrieved per query
• Vector Dimensions: {info.get('vector_dimensions', 0)}                • PCA Components: {info.get('pca_components', 3)}"""


def create_sample_questions_text():
    """
    Create formatted text with sample questions.
    
    Returns:
        Formatted string with categorized sample questions
    """
    questions = get_sample_questions()
    
    # Group questions by category (based on the structure from rag_chain.py)
    categories = [
        ("Cross-Document Comparison Questions:", questions[0:3]),
        ("Sustainability Metrics and Impact Analysis:", questions[3:6]),
        ("Policy and Regulatory Questions:", questions[6:8]),
        ("Application & Practical Evaluation:", questions[8:10])
    ]
    
    formatted_text = "Here are sample questions to explore the sustainability knowledge base:\n\n"
    
    for category_name, category_questions in categories:
        formatted_text += f"**{category_name}**\n\n"
        for i, question in enumerate(category_questions, 1):
            formatted_text += f"{i}. {question}\n\n"
        formatted_text += "\n"
    
    return formatted_text


# ============================================================================
# GRADIO INTERFACE
# ============================================================================

def create_gradio_interface(vector_store_manager, rag_manager, initial_plot):
    """
    Create the Gradio interface for the RAG chatbot.
    
    Args:
        vector_store_manager: Vector store manager instance
        rag_manager: RAG conversation manager instance  
        initial_plot: Initial 3D visualization plot
        
    Returns:
        Configured Gradio Blocks interface
    """
    # Create database info text
    db_info_text = create_database_info_text(vector_store_manager)
    sample_questions_text = create_sample_questions_text()
    
    # Create Gradio interface
    with gr.Blocks(title="RAG Sustainability Chatbot") as demo:
        
        # Header with title and sample questions button
        with gr.Row():
            gr.Markdown("#  Sustainability RAG Chatbot with Vector Visualization")
            test_questions_btn = gr.Button("📋 Sample Questions", size="sm", scale=1)
        
        # Main interface layout
        with gr.Row():
            # Left column: Chat interface
            with gr.Column(scale=3):
                query_input = gr.Textbox(
                    placeholder="Ask a question about sustainability... (Press Enter to send)",
                    label="Your Question",
                    lines=1
                )
                
                chatbot = gr.Chatbot(
                    label="💬 Chat History",
                    height=250
                )
                
                chunk_display = gr.Textbox(
                    label="📦 Retrieved Knowledge Chunks",
                    lines=12,
                    max_lines=12,
                    interactive=False
                )
            
            # Right column: Visualization and info
            with gr.Column(scale=3):
                plot_display = gr.Plot(
                    value=initial_plot,
                    label=" Vector Space Visualization"
                )
                
                info_display = gr.Textbox(
                    value=db_info_text,
                    label="ℹ️ Vector Database Details",
                    lines=5,
                    interactive=False
                )
        
        # Sample questions popup (initially hidden)
        with gr.Column(visible=False) as test_questions_popup:
            gr.Markdown("#  Sample Questions to Test the RAG System")
            gr.Markdown("These questions are designed to test different aspects of the knowledge base:")
            
            gr.Textbox(
                value=sample_questions_text,
                label="Sample Questions by Category",
                lines=20,
                max_lines=25,
                interactive=False
            )
            
            with gr.Row():
                close_popup_btn = gr.Button("✖ Close", variant="secondary")
        
        # Event handlers for popup
        def show_test_questions():
            return gr.update(visible=True)
        
        def hide_test_questions():
            return gr.update(visible=False)
        
        test_questions_btn.click(
            fn=show_test_questions,
            outputs=test_questions_popup
        )
        
        close_popup_btn.click(
            fn=hide_test_questions,
            outputs=test_questions_popup
        )
        
        # Chat interaction handler
        def chat_handler(message, history, plot_fig):
            return chat_with_rag(
                message=message,
                history=history,
                plot_fig=plot_fig,
                vector_store_manager=vector_store_manager,
                rag_manager=rag_manager
            )
        
        # Handle chat submission
        query_input.submit(
            fn=chat_handler,
            inputs=[query_input, chatbot, plot_display],
            outputs=[chatbot, chunk_display, plot_display]
        ).then(
            lambda: "",  # Clear input field
            outputs=[query_input]
        )
    
    return demo


# ============================================================================
# MAIN APPLICATION
# ============================================================================

def main():
    """
    Main function to initialize and launch the RAG chatbot application.
    """
    try:
        # Initialize application components
        vector_store_manager, rag_manager, initial_plot = initialize_application()
        
        # Create and launch Gradio interface
        print("🌐 Creating Gradio interface...")
        demo = create_gradio_interface(vector_store_manager, rag_manager, initial_plot)
        
        print("🚀 Launching application...")
        
        # Check if running in Docker (for deployment)
        is_docker = os.getenv('DOCKER_ENV') == 'true'
        
        if is_docker:
            # Docker/production configuration
            demo.launch(
                server_name="0.0.0.0",
                server_port=7860,
                share=False,
                show_error=True,
                inbrowser=False
            )
        else:
            # Local development configuration
            demo.launch(
                inbrowser=True,
                share=False,
                show_error=True
            )
        
    except FileNotFoundError as e:
        print(f"❌ Vector store not found: {e}")
        print("💡 Please run data_prep.py first to create the vector database")
        
    except Exception as e:
        print(f"❌ Error initializing application: {e}")
        raise


if __name__ == "__main__":
    main()
