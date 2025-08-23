# app/llm_clarifier.py
import os, json, re
from typing import Dict, Any, Optional, List
from openai import OpenAI

# 可用环境变量：
#   OPENAI_API_KEY
#   LLM_CLARIFIER_MODEL (默认 gpt-4o-mini)
#   SLOT_MAP_PATH (可选，指向一个 JSON 配置文件；没有也完全可以工作)
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
_DEFAULT_MODEL = os.getenv("LLM_CLARIFIER_MODEL", "gpt-4o-mini")
_SLOT_MAP_PATH = os.getenv("SLOT_MAP_PATH")  # 可留空

def _load_slot_map() -> Optional[Dict[str, Any]]:
    """可选配置：topics -> required_slots/hints；不存在也没关系。"""
    path = _SLOT_MAP_PATH
    if not path:
        # 没配就返回 None；Clarifier 会走“通用策略”
        return None
    try:
        with open(path, "r") as f:
            data = json.load(f)
            return data if isinstance(data, dict) else None
    except Exception:
        return None

SYSTEM_GENERIC = """You are a conservative triage judge for a blood-donor eligibility assistant.
Return JSON ONLY (no markdown) with this schema:
{
  "decision": "answer" | "clarify",
  "missing_slots": [string],   // <= 3 concise asks; empty if decision="answer"
  "reason": string,
  "confidence": number         // 0..1
}

Clarification policy (no guessing):
- Consider ONLY topics explicitly present/affirmed in user's text or context; do NOT invent new topics.
- If the user explicitly NEGATES a topic (e.g., "no travel", "no infection", "don't have X", "none"), treat that topic as NOT APPLICABLE or SATISFIED with "none". DO NOT ask follow-ups for that topic.
- Do NOT ask about topics not mentioned/affirmed by the user. If the user explicitly negates a topic (e.g., ‘no travel’, ‘no other vaccinations’), treat it as satisfied and do not ask follow-ups for that topic.
- Ask to CLARIFY only if essential facts are missing to apply policy for the active topic(s).
- Typical essentials include: exact dates, vaccine name, tattoo studio license, etc.
- Be minimal: at most 3 actionable questions in missing_slots.
- Do NOT answer the medical question here; only judge clarify vs answer.
- Do NOT ask for general policy facts (e.g., waiting periods/deferral lengths, eligibility rules). The assistant will provide those. Only ask for user-specific facts (dates, license, infection, destination, symptoms, etc.).

"""

SYSTEM_WITH_SLOT_MAP_TMPL = """You are a conservative triage judge for a blood-donor eligibility assistant.

Use the SLOT MAP as guidance for what counts as essential, but:
- Consider ONLY topics explicitly present/affirmed by the user.
- If the user explicitly NEGATES a topic (e.g., "no travel", "no infection"), mark it NOT APPLICABLE / SATISFIED and do NOT ask its slots.
- Do NOT ask for general policy facts (e.g., waiting periods/deferral lengths, eligibility rules). The assistant will provide those. Only ask for user-specific facts (dates, license, infection, destination, symptoms, etc.).

Return JSON ONLY (no markdown) with:
{
  "decision": "answer" | "clarify",
  "missing_slots": [string],   // <= {ask_cap} concise asks
  "reason": string,
  "confidence": number
}

SLOT MAP (compressed):
{slot_map}
"""


def _extract_json(text: str) -> Dict[str, Any]:
    if not text:
        return {}
    m = re.search(r"```json\s*(.*?)```", text, re.S | re.I)
    raw = m.group(1) if m else text
    try:
        return json.loads(raw.strip())
    except Exception:
        return {}

def llm_clarify(question: str,
                context: Optional[Dict[str, Any]] = None,
                max_asks: int = 3) -> Dict[str, Any]:
    """
    Decide whether to ANSWER or CLARIFY. If clarify, return up to max_asks concrete questions.
    Returns: {"decision","missing_slots","reason","confidence"}
    """
    if not question or not question.strip():
        return {"decision": "clarify",
                "missing_slots": ["Please provide your question."],
                "reason": "empty input",
                "confidence": 0.0}

    slot_map = _load_slot_map()
    if slot_map:
        ask_cap = int(slot_map.get("policy", {}).get("ask_cap", max_asks))
        ask_cap = min(max_asks, ask_cap) if isinstance(ask_cap, int) and ask_cap > 0 else max_asks
        system = SYSTEM_WITH_SLOT_MAP_TMPL.format(
            slot_map=json.dumps({
                # 只保留必要字段，避免提示过长
                "topics": {
                    k: {
                        "keywords": (v.get("keywords") or [])[:12],
                        "required_slots": (v.get("required_slots") or [])[:6],
                        "slot_hints": v.get("slot_hints") or {}
                    } for k, v in (slot_map.get("topics") or {}).items()
                },
                "policy": slot_map.get("policy") or {}
            }, ensure_ascii=False),
            ask_cap=ask_cap
        )
        cap = ask_cap
    else:
        system = SYSTEM_GENERIC
        cap = max_asks

    user = question.strip()
    if context:
        try:
            user = ("Context:\n" + json.dumps(context, ensure_ascii=False) +
                    "\n\nUser question:\n" + user)
        except Exception:
            user = "User question:\n" + user

    resp = client.chat.completions.create(
        model=_DEFAULT_MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user}
        ],
        temperature=0.0
    )
    data = _extract_json((resp.choices[0].message.content or "").strip()) or {}
    decision = data.get("decision", "answer")
    if decision not in ("answer", "clarify"):
        decision = "answer"
    slots = data.get("missing_slots") or []
    if not isinstance(slots, list):
        slots = []
    reason = (data.get("reason") or "")[:200]
    try:
        conf = float(data.get("confidence") or 0.0)
    except Exception:
        conf = 0.0

    # 限制追问条数
    slots = slots[:cap]
    # 避免出现空 clarify
    if decision == "clarify" and not slots:
        decision = "answer"

    return {
        "decision": decision,
        "missing_slots": slots,
        "reason": reason,
        "confidence": conf
    }
