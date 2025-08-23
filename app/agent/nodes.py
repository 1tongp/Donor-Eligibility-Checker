# app/agent/nodes.py
import os, json, re, logging
from typing import Dict, Any, Optional
from openai import OpenAI

from app.summarise import summarise_donor, compute_eligibility
from app.chat import rag_answer  # 旧项目的 RAG 查询函数
from app.llm_clarifier import llm_clarify  

# guardrails 可选（若你旧项目路径不同，改这里的导入）
try:
    from app.guardrails import red_flag_hit, escalation_message
except Exception:
    # 兜底：没有 guardrails 时，禁用红旗
    def red_flag_hit(text: str) -> bool:
        return False
    def escalation_message() -> str:
        return "Your query may require clinical escalation. Please consult a clinician."

# --- logging + robust JSON tools ---

os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    filename="logs/agent_debug.log",
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger("agent")

def _parse_json_strict(text: str) -> dict:
    """尽量从模型输出中提取 JSON（支持 ```json 块/大括号片段/整段）。失败返回 {}。"""
    if not text:
        return {}
    m = re.search(r"```json\s*(.*?)```", text, flags=re.S | re.I)
    if m:
        try:
            return json.loads(m.group(1).strip())
        except Exception as e:
            log.warning("fenced json parse fail: %s", e)
    try:
        s = text.find("{"); e = text.rfind("}")
        if s != -1 and e != -1 and e > s:
            return json.loads(text[s:e+1])
    except Exception as e:
        log.warning("brace-slice parse fail: %s", e)
    try:
        return json.loads(text)
    except Exception as e:
        log.error("final json parse fail: %s | raw=%r", e, text)
        return {}

def _normalize_decision_dict(d: dict) -> dict:
    """把模型/反思产出的 decision 统一成规范结构与类型。"""
    if not isinstance(d, dict):
        d = {}
    out = {**d}

    # 1) decision 字段 -> 统一成字符串枚举
    label = out.get("decision")
    if isinstance(label, dict):
        # 有的模型会把 decision 当对象返回，尽量抽取其中的字符串
        label = label.get("label") or label.get("status") or ""
    if label is None:
        label = ""
    label_str = str(label).strip()
    l = label_str.lower()

    mapping = {
        "eligible": "Eligible",
        "ok": "Eligible",
        "yes": "Eligible",
        "ineligible": "Ineligible",
        "no": "Ineligible",
        "defer": "Defer",
        "deferred": "Defer",
        "temporary deferral": "Defer",
        "needmoreinfo": "NeedMoreInfo",
        "need_more_info": "NeedMoreInfo",
        "need more info": "NeedMoreInfo",
        "clarify": "NeedMoreInfo",
    }
    normalized = None
    if l in mapping:
        normalized = mapping[l]
    else:
        if ("need" in l and "info" in l) or ("clarify" in l):
            normalized = "NeedMoreInfo"
        elif "defer" in l:
            normalized = "Defer"
        elif "inelig" in l or "not allow" in l or "cannot" in l:
            normalized = "Ineligible"
        elif "elig" in l or "allow" in l or "can donate" in l:
            normalized = "Eligible"
        else:
            normalized = "NeedMoreInfo"

    out["decision"] = normalized

    # 2) 其它字段类型兜底
    try:
        out["confidence"] = float(out.get("confidence") or 0.5)
    except Exception:
        out["confidence"] = 0.5

    out["rationale"] = str(out.get("rationale") or "")

    mf = out.get("missing_fields")
    if not isinstance(mf, list):
        mf = []
    out["missing_fields"] = [str(x) for x in mf][:3]

    sf = out.get("safety_flags")
    if not isinstance(sf, list):
        sf = []
    out["safety_flags"] = [str(x) for x in sf]

    return out


# online OpenAI client
# client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# local LLM client（使用 Ollama 或其他本地模型）
# client = OpenAI(
#     api_key=os.getenv("OPENAI_API_KEY", "ollama"),  # dummy key
#     base_url=os.getenv("OPENAI_BASE_URL", "http://localhost:11434/v1")
# )

# 根据 USE_LOCAL 开关返回 OpenAI 兼容客户端
def make_openai_client() -> OpenAI:
    """
    根据 USE_LOCAL 返回一个 OpenAI 兼容客户端。
    - 本地：指向 Ollama 的 /v1 端点（base_url）
    - 云端：直连 OpenAI
    """
    use_local = os.getenv("USE_LOCAL", "0") == "1"
    if use_local:
        return OpenAI(
            api_key=os.getenv("OPENAI_API_KEY", "ollama"),     # 占位
            base_url=os.getenv("OPENAI_BASE_URL", "http://localhost:11434/v1")
        )
    else:
        return OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

client = make_openai_client()

# 默认模型名也跟随本地/云端切换
DECISION_MODEL = (
    os.getenv("LOCAL_LLM", "qwen2.5:3b")
    if os.getenv("USE_LOCAL", "0") == "1"
    else os.getenv("DECISION_MODEL", "gpt-4o-mini")
)

# ====== Slot Extraction (LLM → structured slots) ======
EXTRACT_MODEL = os.getenv("EXTRACT_MODEL", DECISION_MODEL)

def _deep_merge_slots(base: dict, add: dict) -> dict:
    """深合并：add 覆盖 base 里为 None/空的值；布尔/字符串/日期有值就覆盖；list 去重扩展。"""
    if not isinstance(base, dict): base = {}
    if not isinstance(add, dict):  add = {}
    out = dict(base)
    for k, v in add.items():
        if isinstance(v, dict):
            out[k] = _deep_merge_slots(out.get(k, {}), v)
        elif isinstance(v, list):
            seen = set(out.get(k, []))
            out[k] = list(seen.union(v))
        else:
            if v not in (None, "", []):  # 明确值才覆盖
                out[k] = v
    return out

def extract_slots_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    用 LLM 从本轮 user text (+少量历史) 中抽取结构化槽位，写入 state['slots']。
    支持否定：no/none/don't have 视为 False 或 'none'。
    """
    q = (state.get("question") or "").strip()
    if not q:
        return state

    system = (
        "You are an information extractor for a blood-donor eligibility agent. "
        "From the user's text, extract ONLY facts explicitly stated (including explicit NEGATIONS). "
        "Return JSON with this schema (no markdown):\n"
        "{\n"
        '  "topics_detected": ["vaccine"|"tattoo"|"travel"|"donation"|"medication"|"symptoms", ...],\n'
        '  "slots": {\n'
        '    "vaccine": {"date":"YYYY-MM-DD|null","type":"string|null","other_recent":true|false|null},\n'
        '    "tattoo": {"date":"YYYY-MM-DD|null","licensed":true|false|null,"infection":true|false|null,"type":"string|null},\n'
        '    "travel": {"recent":true|false|null,"destinations":[{"place":"string","start_date":"YYYY-MM-DD|null","end_date":"YYYY-MM-DD|null"}]},\n'
        '    "donation": {"last_donation_date":"YYYY-MM-DD|null"},\n'
        '    "medication": {"name":"string|null","dose":"string|null","last_taken_date":"YYYY-MM-DD|null"},\n'
        '    "symptoms": {"fever":true|false|null,"dizziness":true|false|null,"infection":true|false|null}\n'
        "  }\n"
        "}\n"
        "Rules: Do NOT invent values. If the user explicitly says 'no/none/don't have', set the corresponding boolean to false or the field to 'none'. "
        "Dates must be ISO YYYY-MM-DD when present; else null. "
        "If a topic is not mentioned, omit it or set fields null."
    )

    user_payload = {
        "text": q,
        "history": state.get("history", [])[-5:],
        "known_slots": state.get("slots", {}),
    }

    raw = ""
    try:
        resp = client.chat.completions.create(
            model=EXTRACT_MODEL,
            messages=[{"role":"system","content":system},
                      {"role":"user","content": json.dumps(user_payload, ensure_ascii=False)}],
            temperature=0.0,
            response_format={"type": "json_object"}  # 强制 JSON（不支持时会被下面兜底接住）
        )
        raw = (resp.choices[0].message.content or "").strip()
    except Exception as e:
        # 本地模型/不支持 response_format 时，退化为普通解析
        try:
            resp = client.chat.completions.create(
                model=EXTRACT_MODEL,
                messages=[{"role":"system","content":system},
                          {"role":"user","content": json.dumps(user_payload, ensure_ascii=False)}],
                temperature=0.0
            )
            raw = (resp.choices[0].message.content or "").strip()
        except Exception:
            raw = ""

    data = {}
    if raw:
        try:
            data = json.loads(raw)
        except Exception:
            data = _parse_json_strict(raw)

    slots = (data.get("slots") or {}) if isinstance(data, dict) else {}
    merged = _deep_merge_slots(state.get("slots") or {}, slots)
    state["slots"] = merged
    state["topics"] = (data.get("topics_detected") or []) if isinstance(data, dict) else []
    return state

# ===== Clarify 去噪过滤 =====
# ===== Clarify 去噪过滤 =====
NEG_NO = r"(?:no|none|don't have|do not have|没有|無|无|未|没有过)"
TOPIC_PATTERNS = {
    "vaccine": r"\b(vaccine|booster|shot|接种|疫苗)\b",
    "tattoo": r"\b(tattoo|piercing|microblading|纹身|穿孔)\b",
    "travel": r"\b(travel|trip|journey|旅游|出行)\b",
    "donation": r"\b(last donation|previous donation|上次献血)\b",
}
# ★ 属于“系统应回答的政策类问题”，一概过滤掉
POLICY_ASK_PATTERNS = [
    r"waiting period", r"how long", r"deferral (?:period|time)",
    r"required .* after .* vaccine", r"when .* allowed to donate",
    r"policy\b", r"guideline",
]

def _detect_topics(text: str) -> set:
    ql = (text or "").lower()
    s = set()
    for k, pat in TOPIC_PATTERNS.items():
        if re.search(pat, ql):
            s.add(k)
    return s

def _filter_clarify_slots(missing, q: str, donor: dict, *, slots: dict = None, topics: list = None) -> list:
    """去掉：已给出的信息、否定过/未提及的话题、政策型问题、或 donor/slots 已有的字段。"""
    ql = (q or "").lower()
    slots = slots or {}
    topics = set(topics or []) or _detect_topics(q)

    # 已给疫苗信息（文本或 slots）
    iso_date_in_text = bool(re.search(r"\b\d{4}-\d{2}-\d{2}\b", q))
    vax = slots.get("vaccine", {}) if isinstance(slots, dict) else {}
    has_vax_date = bool(vax.get("date")) or iso_date_in_text
    has_vax_type = bool(vax.get("type")) or bool(re.search(r"\b(covid|booster|pfizer|moderna|mrna|flu|mmr|hep)\b", ql))
    said_no_other_vax = bool(re.search(rf"{NEG_NO}.*(other )?vaccin", ql)) or (vax.get("other_recent") is False)

    # travel：只有明确提到/肯定最近 travel 才算激活
    travel_slots = slots.get("travel", {}) if isinstance(slots, dict) else {}
    travel_mentioned = ("travel" in topics)
    travel_explicit_no = (travel_slots.get("recent") is False) or bool(re.search(rf"{NEG_NO}.*travel", ql))

    # last donation
    last_donation = (donor or {}).get("last_donation_date") or (
        (slots.get("donation", {}) if isinstance(slots, dict) else {}).get("last_donation_date")
    )
    donation_mentioned = ("donation" in topics)

    out = []
    for s in (missing or []):
        sl = str(s).lower()

        # —— 1) 政策类问题（等待期/规则）→ 系统自己回答，不让 Clarifier 追问
        if any(re.search(p, sl) for p in POLICY_ASK_PATTERNS):
            continue

        # —— 2) 已给 date/when 就不再问
        if ("date" in sl or "when" in sl or "actual date" in sl) and has_vax_date:
            continue

        # —— 3) 已给类型且只是“confirm”就不问
        if ("type" in sl or "vaccine" in sl or "booster" in sl) and has_vax_type and "confirm" in sl:
            continue

        # —— 4) 明确“无其他接种”
        if ("other vaccin" in sl or "any other vaccin" in sl) and said_no_other_vax:
            continue

        # —— 5) travel：未提及 或 明确“无” → 不问
        if "travel" in sl:
            if (not travel_mentioned) and (travel_slots.get("recent") is not True):
                continue
            if travel_explicit_no:
                continue

        # —— 6) last donation：只有用户明确提到且我们确实没有值时才问
        if ("last donation" in sl or "previous donation" in sl):
            if (not donation_mentioned) or last_donation:
                continue

        # —— 7) 泛化“medical conditions”/“health conditions”：若用户未提任何症状，不追问
        if ("medical condition" in sl or "health condition" in sl or "any conditions" in sl):
            if not re.search(r"\b(fever|cold|infection|illness|dizz|symptom|pain)\b", ql):
                continue

        out.append(s)
    return out


# ---------- 基础工具 ----------
def _json(obj: Any) -> str:
    """安全 JSON 序列化（支持 date/datetime）。"""
    try:
        return json.dumps(obj, ensure_ascii=False, default=str)
    except Exception:
        return str(obj)

# ---------- 节点定义 ----------
def ingest_input(state: Dict[str, Any]) -> Dict[str, Any]:
    state = state or {}
    q = (state.get("question") or "").strip()
    hist = state.get("history") or []
    if q:
        hist.append(q)
    state["history"] = hist[-6:]  # 只保留最近几轮
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
    donor = state.get("donor") or {}
    donor_id = donor.get("donor_id") if isinstance(donor, dict) else None
    try:
        state["donor_summary"] = summarise_donor(donor_id) if donor_id else ""
    except Exception:
        state["donor_summary"] = ""
    state["precheck"] = compute_eligibility(donor)
    return state



def rag_node(state: Dict[str, Any]) -> Dict[str, Any]:
    q_user = (state.get("question") or "").strip()
    if not q_user:
        q_user = f"Eligibility determination for donor:\n{state.get('donor_summary','')}"
    try:
        text, cites = rag_answer(q_user, state.get("donor_summary",""))
    except TypeError:
        text, cites = rag_answer(q_user)  # 兼容旧签名
    state["retrieved"] = {"text": text, "citations": cites or []}
    return state

def reason_and_decide_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    先走 clarifier（需要补信息就直接返回 NeedMoreInfo），
    再综合 precheck + RAG 让模型给 JSON 决策。
    模型响应强制为 JSON，任何异常统一兜底为合法 decision。
    """
    q = (state.get("question") or "").strip()

    # ---- 1) LLM clarifier gate ----
    if q:
        try:
            judge = llm_clarify(q,context={
                "history": state.get("history", []),
                "slots": state.get("slots", {}),
                "topics": state.get("topics", []),
                "donor_selected": bool(state.get("donor")),
                "has_precheck": bool(state.get("precheck")),
                })
            asks = _filter_clarify_slots(
                judge.get("missing_slots") or [],
                q,
                state.get("donor") or {},
                slots=state.get("slots", {}),
                topics=state.get("topics", []),
                )

        except Exception as e:
            log.error("llm_clarify failed: %s", e)
            judge = {}
        if judge.get("decision") == "clarify" and asks:
            state["decision"] = {
                "decision": "NeedMoreInfo",
                "confidence": min(0.6, float(judge.get("confidence") or 0.5)),
                "rationale": judge.get("reason") or "Missing essential facts for policy application.",
                "missing_fields": asks[:3],
                "safety_flags": []
            }
            state["used_model"] = os.getenv("LLM_CLARIFIER_MODEL", "gpt-4o-mini")
            return state

    # ---- 2) Decision model ----
    system = (
    "You are a donor eligibility agent.\n"
    "Synthesize hard-rule precheck and retrieved handbook evidence.\n"
    "Return STRICT JSON only with keys: decision, confidence (0..1), rationale, "
    "missing_fields (string[]), safety_flags (string[]).\n\n"
    "Rules:\n"
    "- If essential facts are missing (e.g., exact dates, vaccine type, tattoo studio license, travel destination/dates), "
    "  set decision='NeedMoreInfo' and populate missing_fields with 1-3 concrete, actionable questions.\n"
    "- Do NOT assume values that are not given. Prefer asking for clarification over guessing.\n"
    "- If evidence contradicts precheck, explain in rationale and lower confidence.\n"
    "- If question triggers red flags, include a safety message and set safety_flags.\n"
    "- Output pure JSON. No prose outside JSON."
    )
    payload = {
        "donor": state.get("donor", {}),
        "donor_summary": state.get("donor_summary", ""),
        "precheck": state.get("precheck", {}),
        "retrieved": state.get("retrieved", {}),
        "slots": state.get("slots", {}),  
        "user_question": q
    }

    raw = ""
    try:
        resp = client.chat.completions.create(
            model=DECISION_MODEL,  # ← 你上面已经按 USE_LOCAL/云端设置了
            messages=[
                {"role":"system","content": system},
                {"role":"user","content": _json(payload)}
            ],
            temperature=0.0,
            # ☆ 关键：强制要求 JSON（OpenAI 兼容端会支持；本地不支持也会被捕获兜底）
            response_format={"type": "json_object"}
        )
        raw = (resp.choices[0].message.content or "").strip()
    except Exception as e:
        log.error("decision model call failed: %s", e)
        raw = ""

    log.info("decision raw: %s", raw)

    # 稳健解析 + 兜底
    decision = {}
    if raw:
        try:
            decision = json.loads(raw)
        except Exception as e:
            log.warning("strict loads fail: %s | raw=%r", e, raw)
            decision = _parse_json_strict(raw)

    if not isinstance(decision, dict) or not decision:
        decision = {
            "decision": "NeedMoreInfo",
            "confidence": 0.4,
            "rationale": raw or "unparsable output",
            "missing_fields": [],
            "safety_flags": []
        }

    # 补默认键
    decision.setdefault("decision", "NeedMoreInfo")
    decision.setdefault("confidence", 0.5)
    decision.setdefault("rationale", "")
    decision.setdefault("missing_fields", [])
    decision.setdefault("safety_flags", [])

    decision = _normalize_decision_dict(decision) 
    state["decision"] = decision
    state["used_model"] = DECISION_MODEL
    return state 


def self_reflect_node(state: Dict[str, Any]) -> Dict[str, Any]:
    reflect_sys = (
        "You are a strict validator. Validate and, if needed, correct the decision JSON. "
        "Output JSON only with the same schema."
    )
    payload = {
        "decision": state.get("decision", {}),
        "donor_summary": state.get("donor_summary",""),
        "precheck": state.get("precheck", {}),
        "retrieved": state.get("retrieved", {})
    }
    raw = ""
    try:
        resp = client.chat.completions.create(
            model=DECISION_MODEL,
            messages=[
                {"role":"system","content": reflect_sys},
                {"role":"user","content": _json(payload)}
            ],
            temperature=0.0,
            response_format={"type": "json_object"}
        )
        raw = (resp.choices[0].message.content or "").strip()
    except Exception as e:
        log.error("reflect model call failed: %s", e)
        raw = ""

    log.info("reflect raw: %s", raw)

    fixed = {}
    if raw:
        try:
            fixed = json.loads(raw)
        except Exception as e:
            log.warning("reflect loads fail: %s | raw=%r", e, raw)
            fixed = _parse_json_strict(raw)

    if isinstance(fixed, dict) and fixed:
        merged = state.get("decision", {}).copy()
        merged.update(fixed)
        merged.setdefault("decision", "NeedMoreInfo")
        merged.setdefault("confidence", 0.5)
        merged.setdefault("rationale", "")
        merged.setdefault("missing_fields", [])
        merged.setdefault("safety_flags", [])
        merged = _normalize_decision_dict(merged)  
        state["decision"] = merged

    return state


def explain_node(state: Dict[str, Any]) -> Dict[str, Any]:
    out = _normalize_decision_dict(state.get("decision", {}) or {})   # 归一化
    out["used_model"] = state.get("used_model")
    out.setdefault("rule_citations", [])

    ret = state.get("retrieved")
    if isinstance(ret, dict) and ret.get("citations"):
        for c in ret["citations"]:
            if isinstance(c, dict) and c.get("doc_id"):
                out["rule_citations"].append({"doc_id": c["doc_id"], "text": ""})
            elif isinstance(c, str):
                out["rule_citations"].append({"doc_id": c, "text": ""})

    if "final_status" not in out:
        ds = out.get("decision") or ""            
        dsl = ds.lower()
        if dsl in {"eligible","ineligible","defer","needmoreinfo"}:
            out["final_status"] = out["decision"]
        else:
            pre = state.get("precheck")
            if isinstance(pre, (list, tuple)) and pre:
                out["final_status"] = str(pre[0])
            elif isinstance(pre, dict) and "status" in pre:
                out["final_status"] = pre["status"]

    state["decision"] = out
    return state

