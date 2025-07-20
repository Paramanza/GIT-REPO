import os
import numpy as np
import pandas as pd
from dotenv import load_dotenv
import gradio as gr
import plotly.graph_objects as go
from sklearn.decomposition import PCA
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_core.callbacks import StdOutCallbackHandler

MODEL = "gpt-4o-mini"
db_name = "vector_db"
K_FACTOR = 25

load_dotenv()
os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY', 'your-key-if-not-using-env')

embeddings = OpenAIEmbeddings()
vectorstore = Chroma(persist_directory=db_name, embedding_function=embeddings)

collection = vectorstore._collection
result = collection.get(include=['embeddings', 'documents', 'metadatas'])

vectors = np.array(result['embeddings'])
doc_texts = result['documents']
metadatas = result['metadatas']
doc_types = [metadata['doc_type'] for metadata in metadatas]

pca = PCA(n_components=3)
reduced_vectors = pca.fit_transform(vectors)

llm = ChatOpenAI(temperature=0.7, model_name=MODEL)
memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True, output_key='answer')
retriever = vectorstore.as_retriever(search_kwargs={"k": K_FACTOR})
conversation_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=retriever,
    memory=memory,
    return_source_documents=True,
    callbacks=[StdOutCallbackHandler()],
    output_key="answer",
)

def create_initial_plot():
    unique_doc_types = sorted(set(doc_types))
    hover_texts = [doc[:120].replace('\n', ' ') + "..." for doc in doc_texts]
    df = pd.DataFrame({
        'x': reduced_vectors[:, 0],
        'y': reduced_vectors[:, 1],
        'z': reduced_vectors[:, 2],
        'doc_type': doc_types,
        'text': hover_texts,
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
                marker=dict(size=3, opacity=0.7),
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
        legend=dict(x=0.01, y=0.99, bgcolor='rgba(0,0,0,0)', font=dict(size=10)),
    )
    return go.Figure(data=traces, layout=layout)

def create_query_plot(query_text, source_documents):
    query_vector = embeddings.embed_query(query_text)
    reduced_query = pca.transform([query_vector])[0]
    source_texts = set(doc.page_content for doc in source_documents)
    hover_texts = [doc[:120].replace('\n', ' ') + "..." for doc in doc_texts]
    unique_doc_types = sorted(set(doc_types))
    df = pd.DataFrame({
        'x': reduced_vectors[:, 0],
        'y': reduced_vectors[:, 1],
        'z': reduced_vectors[:, 2],
        'doc_type': doc_types,
        'text': hover_texts,
        'is_retrieved': [doc in source_texts for doc in doc_texts],
    })
    traces = []
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

def chat_with_rag(message, history, plot_fig):
    if not message.strip():
        return history, "", plot_fig
    result = conversation_chain.invoke({"question": message})
    answer = result["answer"]
    source_documents = result.get("source_documents", [])
    history = history + [[message, answer]]
    chunk_texts = []
    for i, doc in enumerate(source_documents):
        chunk_text = f"**Chunk {i+1}** (from {doc.metadata.get('doc_type', 'unknown')}):\n"
        chunk_text += doc.page_content[:500] + "..." if len(doc.page_content) > 500 else doc.page_content
        chunk_texts.append(chunk_text)
    combined_chunks = "\n\n" + "="*50 + "\n\n".join(chunk_texts) if chunk_texts else "No chunks retrieved."
    updated_plot = create_query_plot(message, source_documents)
    return history, combined_chunks, updated_plot

fig = create_initial_plot()
rag_info_compact = f"""**RAG Vector Database Info:**\n• Powered by Chroma                                    • Documents: {len(doc_texts)}\n• Chunk Size: 1600 tokens                            • Total Chunks: {len(doc_texts)}\n• Chunk Overlap: 400 tokens                       • K Factor: 25 retrieved per query"""

with gr.Blocks(title="RAG Sustainability Chatbot") as demo_optimized:
    with gr.Row():
        gr.Markdown("#  Sustainability RAG Chatbot with Vector Visualization")
        test_questions_btn = gr.Button("Sample Questions", size="sm", scale=1)
    with gr.Row():
        with gr.Column(scale=3):
            query_input = gr.Textbox(placeholder="Ask a question about sustainability... (Press Enter to send)", label="Your Question", lines=1)
            chatbot = gr.Chatbot(label="💬 Chat History", height=250)
            chunk_display = gr.Textbox(label="📦 Retrieved Knowledge Chunks", lines=12, max_lines=12, interactive=False)
        with gr.Column(scale=3):
            plot_display = gr.Plot(value=fig, label=" Vector Space Visualization")
            info_display = gr.Textbox(value=rag_info_compact, label=" Vector Database Details", lines=5, interactive=False)
    with gr.Column(visible=False) as test_questions_popup:
        gr.Markdown("# Sample Questions to Test the RAG System")
        gr.Markdown("Here are 10 detailed questions to explore the sustainability knowledge base:")
        test_questions_text = """"""
        gr.Textbox(value=test_questions_text, label="Test Questions", lines=15, max_lines=20, interactive=False)
        with gr.Row():
            close_popup_btn = gr.Button("Close", variant="secondary")
    def show_test_questions():
        return gr.update(visible=True)
    def hide_test_questions():
        return gr.update(visible=False)
    test_questions_btn.click(fn=show_test_questions, outputs=test_questions_popup)
    close_popup_btn.click(fn=hide_test_questions, outputs=test_questions_popup)
    query_input.submit(fn=chat_with_rag, inputs=[query_input, chatbot, plot_display], outputs=[chatbot, chunk_display, plot_display]).then(lambda: "", outputs=[query_input])

demo_optimized.launch(inbrowser=True, share=False)
