# app/guardrails.py
from pathlib import Path
import yaml, re

CFG_PATH = Path("config/guardrails.yaml")

def load_config(path: Path = CFG_PATH) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

CFG = load_config()

def red_flag_hit(text: str) -> bool:
    pats = CFG.get("red_flag_patterns", [])
    for p in pats:
        if re.search(rf"\b{re.escape(p)}\b", text, re.IGNORECASE):
            return True
    return False

def escalation_message() -> str:
    return CFG.get("escalation_message", "Please seek professional medical care.")

def generic_refusal() -> str:
    return CFG.get("generic_refusal", "I can only provide general information.")


import re
from datetime import datetime

_BRACKET_TOKEN = "__BRACKET_BLOCK_%d__"

def _protect_brackets(text: str):
    blocks, out, i = [], [], 0
    for m in re.finditer(r"\[[^\]]+\]", text):
        out.append(text[i:m.start()])
        token = _BRACKET_TOKEN % len(blocks)
        out.append(token)
        blocks.append((token, m.group(0)))
        i = m.end()
    out.append(text[i:])
    return "".join(out), blocks

def _restore_brackets(text: str, blocks):
    for token, val in blocks:
        text = text.replace(token, val)
    return text

def redact_pii(text: str, level: str = "standard") -> str:
    """
    level: "off" | "standard" | "strict"
    - off: 不做脱敏
    - standard: 常见 PII（邮箱/电话/日期/DonorID）+ 仅在“自报姓名”场景脱敏姓名
    - strict: 在 standard 基础上，额外脱敏“看起来像姓名的首字母大写双词”
    - off: No redaction.
    - standard: Common PII (email/phone/date/DonorID) + redact names only in explicit self-introduction contexts.
    - strict: Inherits from standard, additionally redacts any capitalized two-word sequences that look like names.
    """
    if not text or level == "off":
        return text

    # 1) 保护方括号（避免把 [S6]、[F1]、[FAQ] 这些当成要脱敏的目标）
    working, blocks = _protect_brackets(text)

    # 2) 邮箱
    working = re.sub(r"[\w\.-]+@[\w\.-]+", "[REDACTED_EMAIL]", working)

    # 3) 电话（至少 8 个数字的号码）
    def _redact_phone(m):
        digits = re.sub(r"\D", "", m.group(0))
        return "[REDACTED_PHONE]" if len(digits) >= 8 else m.group(0)
    working = re.sub(r"\+?\d[\d\s\-\(\)]{7,}", _redact_phone, working)

    # 4) Donor ID（如 D12345）
    working = re.sub(r"\bD\d{3,8}\b", "[REDACTED_DONOR_ID]", working)

    # 5) 日期（常见格式；可按需再扩展）
    working = re.sub(r"\b\d{1,2}/\d{1,2}/\d{2,4}\b", "[REDACTED_DATE]", working)
    working = re.sub(r"\b\d{4}-\d{1,2}-\d{1,2}\b", "[REDACTED_DATE]", working)

    # 6) 姓名（仅“自报姓名”场景）
    name_prompts = r"(?:my name is|i am|i'm|name\s*:)"
    working = re.sub(
        rf"(?i)\b{name_prompts}\s+([A-Z][a-z]{{2,}}\s+[A-Z][a-z]{{2,}})\b",
        lambda m: m.group(0).replace(m.group(1), "[REDACTED_NAME]"),
        working
    )

    # 7) 严格模式：额外脱敏“看起来像姓名”的首字母大写双词，但避免句首普通词
    if level == "strict":
        def _maybe_name(m):
            # 避免误伤全大写/全小写、长度过短
            first, last = m.group(1), m.group(2)
            if len(first) < 3 or len(last) < 3:
                return m.group(0)
            return "[REDACTED_NAME]"
        working = re.sub(r"\b([A-Z][a-z]{2,})\s+([A-Z][a-z]{2,})\b", _maybe_name, working)

    # 8) 恢复方括号
    working = _restore_brackets(working, blocks)
    return working


# Prompt injection detection
def looks_like_prompt_injection(text: str) -> bool:
    flags = [
        r"ignore (previous|prior) (instructions|rules)",
        r"reveal (system|hidden) prompt",
        r"show (the )?(full|entire) (document|policy)",
        r"print (all )?context",
        r"exfiltrate|leak|bypass (guardrails|safety)",
        r"\bbase64\b|curl\s+http"
    ]
    for p in flags:
        if re.search(p, text, re.IGNORECASE):
            return True
    return False

def prompt_injection_refusal() -> str:
    return "I can’t comply with that request. I will answer based only on allowed policy summaries and won’t reveal internal prompts or full documents."
