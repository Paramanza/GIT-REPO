import os
import glob
import csv
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

MODEL = "gpt-4o-mini"
db_name = "vector_db"
CHUNK_SIZE = 1600
CHUNK_OVERLAP = 400

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
    for chunk in chunks:
        source_path = chunk.metadata.get('source', '')
        for folder_path, doc_type in folder_mapping.items():
            if folder_path in source_path:
                chunk.metadata["doc_type"] = doc_type
                break
        else:
            chunk.metadata["doc_type"] = "unknown"
    return chunks

def build_vector_store():
    print("Setting up vector database...")
    load_dotenv()
    os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY', 'your-key-if-not-using-env')

    folders = glob.glob("knowledge-base/*")
    documents = []
    folder_mapping = {}
    for folder in folders:
        doc_type = os.path.basename(folder)
        folder_mapping[folder] = doc_type
        loader = DirectoryLoader(folder, glob="**/*.txt", loader_cls=SmartTextLoader)
        documents.extend(loader.load())

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ".", " "]
    )
    chunks = text_splitter.split_documents(documents)
    chunks = add_metadata_to_chunks(chunks, folder_mapping)
    print(f"Created {len(chunks)} chunks from {len(documents)} documents")

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

    embeddings = OpenAIEmbeddings()
    if os.path.exists(db_name):
        Chroma(persist_directory=db_name, embedding_function=embeddings).delete_collection()
    Chroma.from_documents(chunks, embedding=embeddings, persist_directory=db_name)
    print("Vector store created")

if __name__ == "__main__":
    build_vector_store()
