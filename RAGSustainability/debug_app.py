"""
Minimal debug version of the RAG app to test Gradio binding.
This helps isolate the server binding issue without loading the full vector store.
"""

import os
import gradio as gr
from dotenv import load_dotenv

def create_simple_interface():
    """Create a minimal interface for testing."""
    def simple_chat(message):
        return f"Echo: {message} (Server is working!)"
    
    with gr.Blocks(title="Debug RAG App") as demo:
        gr.Markdown("# 🔧 Debug RAG App - Server Binding Test")
        
        with gr.Row():
            with gr.Column():
                input_box = gr.Textbox(
                    placeholder="Type something to test the server...",
                    label="Test Input"
                )
                output_box = gr.Textbox(label="Echo Output")
        
        input_box.submit(
            fn=simple_chat,
            inputs=[input_box],
            outputs=[output_box]
        )
    
    return demo

def main():
    """Main debug function."""
    print("🔧 RAG Debug App - Testing Server Binding")
    print("=" * 50)
    
    # Load environment
    load_dotenv()
    
    # Environment detection (same logic as main app)
    is_docker = (
        os.getenv('DOCKER_ENV') == 'true' or 
        os.path.exists('/.dockerenv') or 
        os.getenv('FLY_APP_NAME') is not None
    )
    
    print(f"🔍 Environment detection:")
    print(f"   DOCKER_ENV = {os.getenv('DOCKER_ENV')}")
    print(f"   /.dockerenv exists = {os.path.exists('/.dockerenv')}")
    print(f"   FLY_APP_NAME = {os.getenv('FLY_APP_NAME')}")
    print(f"   is_docker = {is_docker}")
    
    # Create simple interface
    demo = create_simple_interface()
    
    print("\n🚀 Launching debug interface...")
    
    if is_docker:
        print("🐳 Docker mode - binding to 0.0.0.0:7860")
        print("🌐 Should be accessible via Fly.io proxy")
        
        demo.launch(
            server_name="0.0.0.0",
            server_port=7860,
            share=False,
            show_error=True,
            inbrowser=False,
            debug=True,  # Enable debug mode
            quiet=False
        )
    else:
        print("💻 Local mode")
        demo.launch(
            inbrowser=True,
            share=False,
            show_error=True
        )

if __name__ == "__main__":
    main() 