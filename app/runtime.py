# Control the LlamaIndex settings based on environment variables
# This allows for easy switching between local and online LLM/embedding models without code changes.
import os
from llama_index.core import Settings
from dotenv import load_dotenv
load_dotenv()

def apply_llamaindex_settings() -> str:
    """
    根据 USE_LOCAL 开关配置 LlamaIndex 的 LLM 与 Embedding。
    返回 INDEX_DIR（索引持久化路径），以便各处统一引用。
    """
    use_local = os.getenv("USE_LOCAL", "0") == "1" 

    if use_local:
        # 本地：HuggingFace + Ollama
        from llama_index.llms.ollama import Ollama
        from llama_index.embeddings.huggingface import HuggingFaceEmbedding

        llm_model = os.getenv("LOCAL_LLM", "qwen2.5:3b")
        emb_model = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

        Settings.llm = Ollama(model=llm_model, request_timeout=60.0)
        Settings.embed_model = HuggingFaceEmbedding(model_name=emb_model)

        return os.getenv("INDEX_DIR", "index/faiss") # 默认索引目录
    else:
        # 云端：OpenAI
        from llama_index.llms.openai import OpenAI
        from llama_index.embeddings.openai import OpenAIEmbedding

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY is required when USE_LOCAL=0")

        openai_model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        openai_embed = os.getenv("OPENAI_EMBED", "text-embedding-3-small")
        temperature = float(os.getenv("OPENAI_TEMPERATURE", "0.2"))

        Settings.llm = OpenAI(model=openai_model, temperature=temperature, api_key=api_key)
        Settings.embed_model = OpenAIEmbedding(model=openai_embed, api_key=api_key)

        return os.getenv("INDEX_DIR", "index/faiss")  # 默认索引目录
