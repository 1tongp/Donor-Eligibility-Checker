import os
from dotenv import load_dotenv
# app/chat.py
from llama_index.core import StorageContext, load_index_from_storage, Settings
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from guardrails import red_flag_hit, escalation_message, generic_refusal, looks_like_prompt_injection, prompt_injection_refusal
from typing import List, Tuple
def _format_citation(node) -> str:
    meta = getattr(node, 'metadata', {}) or {}
    file_name = meta.get('file_name') or meta.get('source') or 'policy_docs'
    # Try common header metadata keys that LlamaIndex may set for Markdown
    heading = meta.get('section') or meta.get('Header') or meta.get('header') or meta.get('title') or ''
    label = file_name
    if heading:
        label += f" — {heading}"
    return label

# Load .env
load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY not found. Set it in .env or your environment.")
INDEX_DIR = "index/faiss"

POLICY = """Policy: Provide general information only. No diagnosis or treatment recommendations.
If a question suggests serious symptoms, advise seeking medical care. Include citations like [F1] or [S6] where relevant.
"""

def _get_query_engine(top_k: int = 6):
    Settings.llm = OpenAI(model="gpt-4o-mini", temperature=0.2, api_key=api_key)
    Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small", api_key=api_key)
    storage = StorageContext.from_defaults(persist_dir=INDEX_DIR)
    index = load_index_from_storage(storage)
    return index.as_query_engine(similarity_top_k=top_k)

def rag_answer(user_q: str, donor_facts: str | None = None, top_k: int = 6):
    if looks_like_prompt_injection(user_q or ''):
        return prompt_injection_refusal(), ['BLOCKED_PROMPT_INJECTION']
    # Guardrails: block/route severe symptom queries
    if red_flag_hit(user_q):
        return escalation_message(), []
    base = POLICY + "\n"
    if donor_facts:
        base += "Donor facts:\n" + donor_facts + "\n"
    base += "Question:\n" + user_q
    qe = _get_query_engine(top_k=top_k)
    resp = qe.query(base)
    # LlamaIndex response keeps source_nodes; expose doc ids for simple citations
    cites = []  # formatted citations
    try:
        for n in resp.source_nodes:
            doc_id = getattr(n.node, "doc_id", None) or n.node.get_doc_id()
            if doc_id:
                cites.append(doc_id)
    except Exception:
        pass
    return str(resp), cites

if __name__ == "__main__":
    print(rag_answer("Can I donate if I had a tattoo last month?"))
