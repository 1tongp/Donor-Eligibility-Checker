# app/agent/nodes.py
import os, json
from typing import Dict, Any, Optional
from openai import OpenAI

# —— 旧项目模块：保持这些 import 路径与旧项目一致 ——
from app.summarise import summarise_donor, compute_eligibility
from app.chat import rag_answer  # 旧项目的 RAG 查询函数

# guardrails 可选（若你旧项目路径不同，改这里的导入）
try:
    from app.guardrails import red_flag_hit, escalation_message
except Exception:
    # 兜底：没有 guardrails 时，禁用红旗
    def red_flag_hit(text: str) -> bool:
        return False
    def escalation_message() -> str:
        return "Your query may require clinical escalation. Please consult a clinician."

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ---------- 基础工具 ----------
def _json(obj: Any) -> str:
    """安全 JSON 序列化（支持 date/datetime）。"""
    try:
        return json.dumps(obj, ensure_ascii=False, default=str)
    except Exception:
        return str(obj)

# ---------- 节点定义 ----------
def ingest_input(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    state 期望结构：
    {
      "donor": { ... donor fields ... },
      "question": "可选，自然语言问题",
      "meta": {... 可选元数据 ...}
    }
    """
    # 这里可做字段标准化/缺省填充
    return state

def guardrails_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    命中红旗（自残/急症等）时直接出安全话术，阻断后续流程。
    """
    text = (state.get("question") or "") + " " + _json(state.get("donor") or {})
    if red_flag_hit(text):
        state["decision"] = {
            "decision": "NeedMoreInfo",
            "confidence": 0.95,
            "rationale": escalation_message(),
            "missing_fields": [],
            "safety_flags": ["red_flag_detected"]
        }
        state["blocked"] = True
    else:
        state["blocked"] = False
    return state

def precheck_rule_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    使用旧项目的硬/半硬规则预判（compute_eligibility）。
    该函数通常返回结构化结论或标签，我们直接挂到 state 里。
    """
    donor = state.get("donor") or {}
    # summarise 可用于后续提示词
    state["donor_summary"] = summarise_donor(donor)
    # 重要：compute_eligibility 的返回结构以你旧项目为准
    state["precheck"] = compute_eligibility(donor)
    return state

def rag_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    使用旧项目的 RAG 引擎（rag_answer）对“question”或“隐式问题”检索证据。
    你也可以把 donor_summary 拼进去作为 query 上下文。
    """
    q_user = (state.get("question") or "").strip()
    # 典型做法：如果没有明确 question，也可以构造一个内部 query 提示：
    if not q_user:
        q_user = f"Eligibility determination context for donor: {state.get('donor_summary','')}"
    rag_resp = rag_answer(q_user)  # 旧项目返回格式以你的实现为准
    state["retrieved"] = rag_resp
    return state

def reason_and_decide_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    归纳推理节点：
    把 donor 摘要、rule 预判、RAG 证据拼成一个“决策提示词”，
    让模型输出严格 JSON（Eligible/Defer/NeedMoreInfo + 解释 + 置信度 + 缺失字段）。
    """
    system = (
        "You are a donor eligibility agent. "
        "Synthesize hard-rule precheck and retrieved handbook evidence. "
        "Return STRICT JSON with keys: decision, confidence (0..1), rationale (bilingual), "
        "missing_fields (string[]), safety_flags (string[])."
    )

    payload = {
        "donor": state.get("donor", {}),
        "donor_summary": state.get("donor_summary", ""),
        "precheck": state.get("precheck", {}),
        "retrieved": state.get("retrieved", {}),
        "user_question": state.get("question", "")
    }

    resp = client.chat.completions.create(
        model=os.getenv("DECISION_MODEL", "gpt-4o-mini"),
        messages=[
            {"role":"system","content": system},
            {"role":"user","content": _json(payload)}
        ],
        temperature=0.1
    )
    text = (resp.choices[0].message.content or "").strip()
    try:
        decision = json.loads(text)
        # 兜底：字段缺失时补齐
        decision.setdefault("decision", "NeedMoreInfo")
        decision.setdefault("confidence", 0.5)
        decision.setdefault("rationale", "")
        decision.setdefault("missing_fields", [])
        decision.setdefault("safety_flags", [])
    except Exception:
        decision = {
            "decision": "NeedMoreInfo",
            "confidence": 0.4,
            "rationale": text,
            "missing_fields": [],
            "safety_flags": []
        }
    state["decision"] = decision
    state["used_model"] = os.getenv("DECISION_MODEL", "gpt-4o-mini")
    return state

def self_reflect_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    自反思：再次让模型检查 JSON 是否自洽、是否与证据矛盾、是否缺字段；如有问题则修正。
    """
    decision = state.get("decision", {})
    reflect_sys = (
        "You are a strict validator. If the decision JSON contradicts the evidence or "
        "misses necessary fields, fix it and output corrected JSON only."
    )
    reflect_payload = {
        "decision": decision,
        "donor_summary": state.get("donor_summary",""),
        "precheck": state.get("precheck", {}),
        "retrieved": state.get("retrieved", {})
    }
    resp = client.chat.completions.create(
        model=os.getenv("DECISION_MODEL", "gpt-4o-mini"),
        messages=[
            {"role":"system","content": reflect_sys},
            {"role":"user","content": _json(reflect_payload)}
        ],
        temperature=0.0
    )
    text = (resp.choices[0].message.content or "").strip()
    try:
        fixed = json.loads(text)
        # 合并/覆盖：若输出仍不是 JSON，就忽略
        if isinstance(fixed, dict):
            state["decision"] = fixed
    except Exception:
        pass
    return state

def explain_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    收尾输出：把引用信息、模型名等补齐；可在此统一结构化返回。
    """
    out = state.get("decision", {})
    out["used_model"] = state.get("used_model")
    # 如果你在 rag_answer 里有明确的 citations/source_nodes，可以放到 rule_citations
    if "rule_citations" not in out:
        out["rule_citations"] = []
    # 可把 RAG 的来源附上（按你旧项目的返回结构适配）
    if isinstance(state.get("retrieved"), dict) and "citations" in state["retrieved"]:
        for c in state["retrieved"]["citations"]:
            out["rule_citations"].append({"doc_id": c, "text": ""})
    return out
