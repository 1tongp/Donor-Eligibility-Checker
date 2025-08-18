# app/app_gradio.py
import os
import json
import difflib
import hashlib
from datetime import datetime
from pathlib import Path
import re

from dotenv import load_dotenv
import gradio as gr
import pandas as pd

from build_index import build_index
from summarise import summarise_donor
from chat import rag_answer
from guardrails import redact_pii, looks_like_prompt_injection, prompt_injection_refusal

# ---------- Config & bootstrap ----------
load_dotenv()
API_KEY = os.getenv("OPENAI_API_KEY")
if not API_KEY:
    raise ValueError("OPENAI_API_KEY not found. Set it in .env or your environment.")

INDEX_DIR = "index/faiss"
LOG_DIR = "logs"
LOG_FILE = os.path.join(LOG_DIR, "qa_logs.jsonl")
os.makedirs(LOG_DIR, exist_ok=True)

def _ensure_index():
    p = Path(INDEX_DIR)
    if not p.exists() or not any(p.iterdir()):
        try:
            print("Index missing. Building once ...")
            p.mkdir(parents=True, exist_ok=True)
            build_index()
        except Exception as e:
            print("Auto-build index failed:", e)

# Data
donors = pd.read_csv("data/donors.csv")
donor_ids = donors["donor_id"].tolist()
with open("data/faqs.json", "r") as f:
    FAQS = json.load(f)

# ---------- Utilities ----------
def _audit_log(record: dict):
    try:
        with open(LOG_FILE, "a") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception as e:
        print("Audit log failed:", e)

def _hash(s: str) -> str:
    return hashlib.sha256((s or "").encode()).hexdigest()[:12]

def _match_faq(question: str, threshold: float = 0.72):
    if not question or not FAQS:
        return None
    candidates = [faq["q"] for faq in FAQS]
    best = difflib.get_close_matches(question, candidates, n=1, cutoff=0.0)
    if not best:
        return None
    q_best = best[0]
    ratio = difflib.SequenceMatcher(a=question.lower(), b=q_best.lower()).ratio()
    if ratio >= threshold:
        faq = next(f for f in FAQS if f["q"] == q_best)
        return {
            "q": faq["q"],
            "a": faq["a"],
            "source": faq.get("source", "FAQ"),
            "score": ratio,
        }
    return None

# ---------- App logic ----------
def _extract_json_block(text: str):
    """
    从 LLM 返回的字符串里提取 ```json fenced block
    如果没有 fenced block，就尝试直接 json.loads
    """
    if not text:
        return None
    # 提取 ```json ... ```
    match = re.search(r"```json(.*?)```", text, re.DOTALL | re.IGNORECASE)
    if match:
        try:
            return json.loads(match.group(1).strip())
        except Exception:
            return None
    # fallback: 直接尝试解析整个字符串
    try:
        return json.loads(text.strip())
    except Exception:
        return None
    
def ui_summarise(did: str):
    _ensure_index()
    return summarise_donor(did)

def ui_chat(mode: str, qtype: str, donor_id: str, question: str, redact_level: str = "standard"):
    _ensure_index()
    ts = datetime.utcnow().isoformat() + "Z"
    facts = ""
    donor_row = None

    # -------- Donor basic facts（仍然保留，给 RAG 更多可读上下文） --------
    if mode == "Donor-specific" and donor_id:
        donor_row = donors[donors["donor_id"] == donor_id]
        if len(donor_row) > 0:
            r = donor_row.iloc[0]
            facts = (
                f"sex:{r.get('sex')} age:{r.get('age')} "
                f"hb:{r.get('hb_g_dl')} "
                f"bp:{r.get('systolic_bp')}/{r.get('diastolic_bp')} "
                f"bmi:{r.get('bmi')} "
                f"flags:{r.get('questionnaire_flags')}"
            )

    # -------- Prompt injection defense --------
    if looks_like_prompt_injection(question or ""):
        ans = prompt_injection_refusal()
        ans = redact_pii(ans, level=redact_level)
        _audit_log({
            "ts": ts, "mode": mode, "qtype": qtype,
            "donor_id": donor_id, "question": question,
            "facts": facts, "answer": ans, "citations": ["BLOCKED_PROMPT_INJECTION"],
        })
        return ans

    # -------- FAQ fast path（不走 LLM，保持省钱与稳定） --------
    if qtype == "FAQ":
        match = _match_faq(question)
        if match:
            ans = f"""{match['a']}

Sources:
- {match['source']}"""
            ans = redact_pii(ans, level=redact_level)
            _audit_log({
                "ts": ts, "mode": mode, "qtype": qtype,
                "donor_id": donor_id, "question": question,
                "facts": facts, "answer_hash": _hash(ans),
                "citations": [match["source"]], "faq_score": match["score"],
            })
            return ans
        # No match → guide user to Freeform
        ans = "I couldn’t find a close FAQ match. Switch to Freeform to search the full policy."
        _audit_log({
            "ts": ts, "mode": mode, "qtype": qtype,
            "donor_id": donor_id, "question": question,
            "facts": facts, "answer": ans, "citations": ["NO_FAQ_MATCH"],
        })
        return ans

    # -------- Freeform → 始终附带 donor 的 JSON summary --------
    donor_json_ctx = ""
    donor_cites = []
    if mode == "Donor-specific" and donor_id:
        try:
            # 兼容包/脚本两种运行方式
            try:
                from app.summarise import summarise_donor as _summary
            except Exception:
                from summarise import summarise_donor as _summary
            summary_text = _summary(donor_id)
            # 尝试提取/解析 JSON（支持 ```json fenced block）
            data = _extract_json_block(summary_text) or {}
            donor_json_ctx = "Donor Summary JSON:\n" + json.dumps(data, ensure_ascii=False)
            donor_cites = data.get("policy_citations") or []
        except Exception:
            # 解析失败也不阻塞，继续仅用 facts
            donor_json_ctx = ""

    # 拼装传给 RAG 的上下文（facts + JSON）
    effective_facts = "\n".join([p for p in [facts, donor_json_ctx] if p]).strip() or None

    ans, cites = rag_answer(question, effective_facts)
    ans = redact_pii(ans, level=redact_level)
    cites = cites or donor_cites or []
    _audit_log({
        "ts": ts, "mode": mode, "qtype": qtype,
        "donor_id": donor_id, "question": question,
        "facts": (effective_facts[:5000] if effective_facts else ""),  # 避免日志过长
        "answer_hash": _hash(ans),
        "citations": cites,
        "routed": "freeform_with_donor_json" if donor_json_ctx else "freeform_no_json"
    })
    if cites:
        ans = ans + "\n\nSources:\n- " + "\n- ".join(cites)
    return ans

# ---------- UI ----------
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# Donor Health Summary & RAG Chat (Local, OpenAI)")
    with gr.Row():
        redact_level = gr.Radio(["off", "standard", "strict"], value="standard", label="Redaction")
        mode = gr.Radio(["General", "Donor-specific"], value="General", label="Mode")
        qtype = gr.Radio(["FAQ", "Freeform"], value="FAQ", label="Query Type")
        did = gr.Dropdown(donor_ids, label="Select Donor", scale=1)
        btn = gr.Button("Generate Summary", scale=0)
    out = gr.Textbox(label="Summary (JSON)", lines=10)
    btn.click(ui_summarise, inputs=did, outputs=out)

    gr.Markdown("## Ask a Question")
    q_inp = gr.Textbox(label="Your question", lines=2, placeholder="e.g., Can I donate if my Hb is 11.8?")
    a = gr.Textbox(label="Answer", lines=8)
    ask_btn = gr.Button("Ask")
    ask_btn.click(ui_chat, inputs=[mode, qtype, did, q_inp, redact_level], outputs=a)

# Build index if needed on import
_ensure_index()

if __name__ == "__main__":
    demo.launch()
