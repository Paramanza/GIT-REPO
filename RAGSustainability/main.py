import os
import glob
from dotenv import load_dotenv
import gradio as gr
from langchain.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.schema import Document
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import numpy as np
import plotly.graph_objects as go
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
import chardet
from langchain_community.document_loaders import DirectoryLoader
import plotly.graph_objs as go
from sklearn.decomposition import PCA
import pandas as pd
import matplotlib.cm as cm
import csv
import plotly.io as pio
from langchain_core.callbacks import StdOutCallbackHandler

# Configuration
MODEL = "gpt-4o-mini"
db_name = "vector_db"
CHUNK_SIZE = 1600
CHUNK_OVERLAP = 400
K_FACTOR = 25

# Load environment variables
load_dotenv()
os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY', 'your-key-if-not-using-env')

# Smart text loader class
class SmartTextLoader(TextLoader):
    def __init__(self, file_path, autodetect_encoding=True):
        if autodetect_encoding:
            with open(file_path, "rb") as f:
                import chardet
                result = chardet.detect(f.read())
                encoding = result['encoding'] or 'utf-8'
        else:
            encoding = 'utf-8'
        super().__init__(file_path, encoding=encoding)

def add_metadata_to_chunks(chunks, folder_mapping):
    """Add metadata to each chunk based on original document source"""
    for chunk in chunks:
        # Get source file path from metadata
        source_path = chunk.metadata.get('source', '')
        # Find which folder this chunk originated from
        for folder_path, doc_type in folder_mapping.items():
            if folder_path in source_path:
                chunk.metadata["doc_type"] = doc_type
                break
        else:
            chunk.metadata["doc_type"] = "unknown"
    return chunks

# ============================================================================
# VECTOR DATABASE SETUP
# ============================================================================

print("Setting up vector database...")

# Load documents from knowledge base folders
folders = glob.glob("knowledge-base/*")
documents = []
folder_mapping = {}

for folder in folders:
    doc_type = os.path.basename(folder)
    folder_mapping[folder] = doc_type
    loader = DirectoryLoader(
        folder,
        glob="**/*.txt",
        loader_cls=SmartTextLoader
    )
    folder_docs = loader.load()
    documents.extend(folder_docs)

# Chunk the documents
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    separators=["\n\n", "\n", ".", " "]
)

chunks = text_splitter.split_documents(documents)

# Apply metadata to each chunk (NOT just documents)
chunks = add_metadata_to_chunks(chunks, folder_mapping)

print(f"Created {len(chunks)} chunks from {len(documents)} documents")

# Save chunk preview to CSV
with open("chunk_preview.csv", "w", encoding="utf-8", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Index", "Doc Type", "Length", "Content"])
    for i, chunk in enumerate(chunks):
        writer.writerow([
            i + 1,
            chunk.metadata.get("doc_type", "unknown"),
            len(chunk.page_content),
            chunk.page_content.replace("\n", " ")
        ])

# Create vector store
embeddings = OpenAIEmbeddings()

if os.path.exists(db_name):
    Chroma(persist_directory=db_name, embedding_function=embeddings).delete_collection()

vectorstore = Chroma.from_documents(documents=chunks, embedding=embeddings, persist_directory=db_name)
print(f"Vector store created with {vectorstore._collection.count()} documents")

# ============================================================================
# PREPARE VECTOR DATA FOR VISUALIZATION
# ============================================================================

collection = vectorstore._collection
result = collection.get(include=['embeddings', 'documents', 'metadatas'])

vectors = np.array(result['embeddings'])
doc_texts = result['documents']
metadatas = result['metadatas']
doc_types = [metadata['doc_type'] for metadata in metadatas]

# Reduce dimensionality for visualization
pca = PCA(n_components=3)
reduced_vectors = pca.fit_transform(vectors)

# ============================================================================
# RAG SETUP
# ============================================================================

# Create LLM and conversation chain
llm = ChatOpenAI(temperature=0.7, model_name=MODEL)
memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True, output_key='answer')
retriever = vectorstore.as_retriever(search_kwargs={"k": K_FACTOR})
conversation_chain = ConversationalRetrievalChain.from_llm(
    llm=llm, 
    retriever=retriever, 
    memory=memory, 
    return_source_documents=True, 
    callbacks=[StdOutCallbackHandler()],
    output_key="answer"
)

# ============================================================================
# PLOTTING FUNCTIONS
# ============================================================================

def create_initial_plot():
    """Create initial 3D plot showing all chunks"""
    unique_doc_types = sorted(set(doc_types))
    hover_texts = [doc[:120].replace('\n', ' ') + "..." for doc in doc_texts]
    
    df = pd.DataFrame({
        'x': reduced_vectors[:, 0],
        'y': reduced_vectors[:, 1],
        'z': reduced_vectors[:, 2],
        'doc_type': doc_types,
        'text': hover_texts
    })
    
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
                marker=dict(size=3, opacity=0.7)
            )
        )
    
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
        legend=dict(
            x=0.01, y=0.99,
            bgcolor='rgba(0,0,0,0)',
            font=dict(size=10)
        )
    )
    
    return go.Figure(data=traces, layout=layout)

def create_query_plot(query_text, source_documents):
    """Create updated plot showing query and retrieved chunks"""
    # Get query embedding and reduce it
    query_vector = embeddings.embed_query(query_text)
    reduced_query = pca.transform([query_vector])[0]
    
    # Identify retrieved chunks
    source_texts = set(doc.page_content for doc in source_documents)
    hover_texts = [doc[:120].replace('\n', ' ') + "..." for doc in doc_texts]
    
    unique_doc_types = sorted(set(doc_types))
    df = pd.DataFrame({
        'x': reduced_vectors[:, 0],
        'y': reduced_vectors[:, 1],
        'z': reduced_vectors[:, 2],
        'doc_type': doc_types,
        'text': hover_texts,
        'is_retrieved': [doc in source_texts for doc in doc_texts]
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
                    marker=dict(size=2, opacity=0.3)
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
                marker=dict(size=6, color='white', opacity=1.0)
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
            marker=dict(size=8, color='yellow', symbol='diamond')
        )
    )
    
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
        legend=dict(
            x=0.01, y=0.99,
            bgcolor='rgba(0,0,0,0)',
            font=dict(size=10)
        )
    )
    
    return go.Figure(data=traces, layout=layout)

# ============================================================================
# CHAT FUNCTION
# ============================================================================

def chat_with_rag(message, history):
    """Handle chat interaction and return updated plot"""
    if not message.strip():
        return history, "", create_initial_plot()
    
    # Get RAG response
    result = conversation_chain.invoke({"question": message})
    answer = result["answer"]
    source_documents = result.get("source_documents", [])
    
    # Update chat history
    history = history + [[message, answer]]
    
    # Extract chunk contents for display
    chunk_texts = []
    for i, doc in enumerate(source_documents):
        chunk_text = f"**Chunk {i+1}** (from {doc.metadata.get('doc_type', 'unknown')}):\n"
        chunk_text += doc.page_content[:500] + "..." if len(doc.page_content) > 500 else doc.page_content
        chunk_texts.append(chunk_text)
    
    combined_chunks = "\n\n" + "="*50 + "\n\n".join(chunk_texts) if chunk_texts else "No chunks retrieved."
    
    # Create updated plot
    updated_plot = create_query_plot(message, source_documents)
    
    return history, combined_chunks, updated_plot

# WIDESCREEN INTERFACE WITH OPTIMIZED LAYOUT
# Based on your red markings - RAG info in 2 columns, adjusted heights for alignment

# Create initial plot
fig = create_initial_plot()

# Create two-column RAG info text (exactly as you marked in red)
rag_info_compact = f"""**RAG Vector Database Info:**
• Powered by Chroma                                    • Documents: {len(documents)}
• Chunk Size: 1600 tokens                            • Total Chunks: {len(chunks)}
• Chunk Overlap: 400 tokens                       • K Factor: 25 retrieved per query"""

def chat_with_rag(message, history, plot_fig):
    """Enhanced chat function that works with the current setup"""
    if not message.strip():
        return history, "", plot_fig
    
    # Get RAG response using existing conversation_chain
    result = conversation_chain.invoke({"question": message})
    answer = result["answer"]
    source_documents = result.get("source_documents", [])
    
    # Update chat history
    history = history + [[message, answer]]
    
    # Format chunk contents for display
    chunk_texts = []
    for i, doc in enumerate(source_documents):
        chunk_text = f"**Chunk {i+1}** (from {doc.metadata.get('doc_type', 'unknown')}):\n"
        chunk_text += doc.page_content[:500] + "..." if len(doc.page_content) > 500 else doc.page_content
        chunk_texts.append(chunk_text)
    
    combined_chunks = "\n\n" + "="*50 + "\n\n".join(chunk_texts) if chunk_texts else "No chunks retrieved."
    
    # Create updated plot with query visualization
    updated_plot = create_query_plot(message, source_documents)
    return history, combined_chunks, updated_plot

print("🎯 Creating widescreen-optimized interface...")

with gr.Blocks(title="RAG Sustainability Chatbot") as demo_optimized:
    gr.Markdown("#  Sustainability RAG Chatbot with Vector Visualization")
    
    with gr.Row():
        # Left column 
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
            
            # INCREASED HEIGHT to align with bottom of info box (as marked in red)
            chunk_display = gr.Textbox(
                label="📦 Retrieved Knowledge Chunks",
                lines=12,  # Increased from 8 to align with reduced info box
                max_lines=12,
                interactive=False
            )
        
        # Right column
        with gr.Column(scale=3):
            plot_display = gr.Plot(
                value=fig,
                label=" Vector Space Visualization"
            )
            
            # REDUCED HEIGHT info box with TWO-COLUMN text (as marked in red)
            info_display = gr.Textbox(
                value=rag_info_compact,
                label=" Vector Database Details",
                lines=5,  # HALVED from 6 to 3 lines as marked in red
                interactive=False
            )
    
    # Handle chat submission
    query_input.submit(
        fn=chat_with_rag,
        inputs=[query_input, chatbot, plot_display],
        outputs=[chatbot, chunk_display, plot_display]
    ).then(
        lambda: "",  # Clear input
        outputs=[query_input]
    )

# Launch the optimized interface
demo_optimized.launch(inbrowser=True, share=False)
