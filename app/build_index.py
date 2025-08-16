# app/build_index.py
import os
from pathlib import Path
from dotenv import load_dotenv
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding

# Load .env
load_dotenv()
API_KEY = os.getenv("OPENAI_API_KEY")
if not API_KEY:
    raise ValueError("OPENAI_API_KEY not found. Set it in .env or your environment.")

DOC_DIR = "data/policy_docs"
INDEX_DIR = "index/faiss"

def _metadata_extractor(file_path: str):
    """Attach basic filename metadata so we can render '文档名 + 段落标题'."""
    return {"file_name": Path(file_path).name}

def build_index():
    # Configure LLM & embeddings
    Settings.llm = OpenAI(model="gpt-4o-mini", temperature=0.1, api_key=API_KEY)
    Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small", api_key=API_KEY)

    # Load docs with metadata
    docs = SimpleDirectoryReader(
        DOC_DIR,
        recursive=True,
        file_metadata=_metadata_extractor
    ).load_data()

    # Build & persist index
    index = VectorStoreIndex.from_documents(docs)
    index.storage_context.persist(persist_dir=INDEX_DIR)
    print(f"✅ Index built and saved to: {INDEX_DIR}")

if __name__ == "__main__":
    os.makedirs(INDEX_DIR, exist_ok=True)
    build_index()
