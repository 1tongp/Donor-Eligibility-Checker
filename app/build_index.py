# app/build_index.py
import os
from pathlib import Path
from dotenv import load_dotenv

from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.core.text_splitter import TokenTextSplitter
from llama_index.core.schema import TextNode

from runtime import apply_llamaindex_settings  # 你的统一 LLM/Embedding 配置入口

# 初始化 LlamaIndex（会根据 USE_LOCAL 等环境变量配置）
INDEX_DIR = apply_llamaindex_settings()

# .env
load_dotenv()
DOC_DIR = "data/policy_docs"


def _metadata_extractor(file_path: str):
    """给每个文档附上文件名，便于引用展示。"""
    return {"file_name": Path(file_path).name}


def build_index():
    if not Path(DOC_DIR).exists():
        raise FileNotFoundError(f"{DOC_DIR} not found. Put your rules/docs under this folder.")

    # 1) 读取文档并附加 file_name 元数据
    docs = SimpleDirectoryReader(
        DOC_DIR,
        recursive=True,
        file_metadata=_metadata_extractor
    ).load_data()

    if not docs:
        print(f"No documents found under {DOC_DIR}")
        return

    # 2) 使用 TokenTextSplitter 手工切分（不依赖 NLTK）
    splitter = TokenTextSplitter(chunk_size=800, chunk_overlap=120)

    nodes = []
    for d in docs:
        chunks = splitter.split_text(d.text or "")
        for idx, chunk in enumerate(chunks):
            md = dict(d.metadata or {})
            # 再兜底一层 file_name
            if "file_name" not in md:
                md["file_name"] = md.get("filename") or md.get("source") or ""
            md["chunk_id"] = idx  # 可选：方便 debug
            nodes.append(TextNode(text=chunk, metadata=md))

    if not nodes:
        print("No nodes split from documents; check your files.")
        return

    # 3) 构建索引并持久化
    index = VectorStoreIndex(nodes)
    os.makedirs(INDEX_DIR, exist_ok=True)
    index.storage_context.persist(persist_dir=INDEX_DIR)
    print(f"✅ Indexed {len(nodes)} nodes from {len(docs)} docs → {INDEX_DIR}")


if __name__ == "__main__":
    build_index()
