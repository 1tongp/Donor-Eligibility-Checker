# app/summarise.py
import os
from dotenv import load_dotenv
import pandas as pd
from typing import Tuple, List
from llama_index.core import StorageContext, load_index_from_storage, Settings
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding

# -------- Eligibility precheck (rule-based) --------
def compute_eligibility(row: dict) -> Tuple[str, List[str]]:
    """
    Simple rule-based precheck for eligibility. Returns (status, reasons).
    status in { 'eligible', 'ineligible', 'require_medical_clearance' }.
    """
    reasons: List[str] = []
    sex = str(row.get('sex', '')).lower()
    hb = float(row.get('hb_g_dl', 0) or 0)
    sys_bp = float(row.get('systolic_bp', 0) or 0)
    dia_bp = float(row.get('diastolic_bp', 0) or 0)
    bmi = float(row.get('bmi', 0) or 0)
    flags = str(row.get('questionnaire_flags', '') or '').lower()

    # Ineligible
    if (sex.startswith('f') and hb < 12.5) or (sex.startswith('m') and hb < 13.0):
        reasons.append(f"Low Hb: {hb} g/dL")
    if sys_bp >= 180 or dia_bp >= 110:
        reasons.append(f"Very high blood pressure: {int(sys_bp)}/{int(dia_bp)} mmHg")

    status = 'eligible'
    if reasons:
        status = 'ineligible'

    # Medical clearance
    med_flags = ['tattoo_3m', 'recent_surgery', 'recent_antibiotics']
    if any(f in flags for f in med_flags) or bmi >= 45:
        if status != 'ineligible':
            status = 'require_medical_clearance'
        reasons.append("Recent risk factor flags or high BMI")

    if not reasons:
        reasons.append('Meets basic precheck thresholds')
    return status, reasons

# -------- Config --------
load_dotenv()
API_KEY = os.getenv("OPENAI_API_KEY")
if not API_KEY:
    raise ValueError("OPENAI_API_KEY not found. Set it in .env or your environment.")

INDEX_DIR = "index/faiss"
DONOR_CSV = "data/donors.csv"

# 强制模型返回严格 JSON（含 eligibility 字段）
SYSTEM_RULES = """You are a clinical information assistant. Return a STRICT JSON object with keys:
- donor_id: string
- vitals: {sex, age, hb_g_dl, blood_pressure, bmi}
- eligibility_status: one of ["eligible","ineligible","require_medical_clearance"]
- eligibility_reasons: array of string
- policy_citations: array of string (human-readable like "eligibility_rules.md — Tattoos")
- summary: string (concise, general information only; no diagnosis/treatment)
If information is missing, keep fields but use empty strings/arrays. Do not add extra keys.
If symptoms are severe, advise seeking medical care.
"""

def _get_query_engine():
    Settings.llm = OpenAI(model="gpt-4o-mini", temperature=0.1, api_key=API_KEY)
    Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small", api_key=API_KEY)
    storage = StorageContext.from_defaults(persist_dir=INDEX_DIR)
    index = load_index_from_storage(storage)
    return index.as_query_engine(similarity_top_k=6)

def summarise_donor(donor_id: str) -> str:
    df = pd.read_csv(DONOR_CSV)
    match = df[df["donor_id"] == donor_id]
    if match.empty:
        return f'{{"donor_id":"{donor_id}","vitals":{{}},"eligibility_status":"","eligibility_reasons":[],"policy_citations":[],"summary":"Donor not found"}}'
    row = match.iloc[0].to_dict()

    facts = f"""
Donor Facts:
- donor_id: {donor_id}
- sex: {row.get('sex')}
- age: {row.get('age')}
- hb_g_dl: {row.get('hb_g_dl')}
- blood_pressure: {row.get('systolic_bp')}/{row.get('diastolic_bp')} mmHg
- bmi: {row.get('bmi')}
- questionnaire_flags: {row.get('questionnaire_flags')}
""".strip()

    status, reasons = compute_eligibility(row)
    precheck = f"Precheck: eligibility_status={status}; reasons={reasons}"

    qe = _get_query_engine()
    prompt = f"""{SYSTEM_RULES}

{facts}
{precheck}

Retrieve the most relevant policy sections and return a STRICT JSON with keys:
donor_id, vitals, eligibility_status, eligibility_reasons, policy_citations, summary.
Set eligibility_status equal to the Precheck unless cited policy clearly overrides it.
policy_citations should be human-readable like "eligibility_rules.md — Tattoos".
"""
    resp = qe.query(prompt)
    return str(resp)

if __name__ == "__main__":
    print(summarise_donor("D1000"))
