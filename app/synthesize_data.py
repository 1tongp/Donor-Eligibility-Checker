# app/synthesize_data.py
"""Generate synthetic donor health data for learning and testing.
Usage:
  python app/synthesize_data.py --n 300 --seed 123 --out data/donors.csv
"""
import argparse, random, math, datetime as dt
from pathlib import Path
import pandas as pd

FLAGS = [
    "recent_travel", "recent_antibiotics", "tattoo_3m",
    "recent_surgery", "none"
]

def gen_row(i:int, start_date:dt.date) -> dict:
    donor_id = f"D{1000+i:04d}"
    sex = random.choice(["M","F"])
    # Basic distributions
    age = max(18, min(70, int(random.gauss(35, 10))))
    hb = round(random.gauss(14 if sex=="M" else 13, 1.1), 1)   # g/dL
    sbp = int(random.gauss(122, 14))
    dbp = int(random.gauss(78, 10))
    bmi = round(random.gauss(24.5, 4.2), 1)
    last_date = start_date + dt.timedelta(days=random.randint(0, 450))
    qflags = random.choice(FLAGS)

    return {
        "donor_id": donor_id,
        "sex": sex,
        "age": age,
        "hb_g_dl": hb,
        "systolic_bp": sbp,
        "diastolic_bp": dbp,
        "bmi": bmi,
        "last_donation_date": last_date.isoformat(),
        "questionnaire_flags": qflags
    }

def inject_edge_cases(df: pd.DataFrame, frac_low_hb=0.08, frac_high_bp=0.06, frac_bmi=0.05) -> pd.DataFrame:
    n = len(df)
    # Low Hb
    idxs = df.sample(max(1, int(n*frac_low_hb))).index
    for idx in idxs:
        df.loc[idx, "hb_g_dl"] = round(random.uniform(10.5, 11.9), 1)
    # High BP
    idxs = df.sample(max(1, int(n*frac_high_bp))).index
    for idx in idxs:
        df.loc[idx, "systolic_bp"] = random.randint(165, 190)
        df.loc[idx, "diastolic_bp"] = random.randint(100, 120)
    # High BMI
    idxs = df.sample(max(1, int(n*frac_bmi))).index
    for idx in idxs:
        df.loc[idx, "bmi"] = round(random.uniform(41, 50), 1)
    return df

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200, help="number of rows")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=Path, default=Path("data/donors.csv"))
    args = ap.parse_args()

    random.seed(args.seed)
    rows = []
    start_date = dt.date(2024, 1, 1)
    for i in range(args.n):
        rows.append(gen_row(i, start_date))
    df = pd.DataFrame(rows)
    df = inject_edge_cases(df)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"Wrote {len(df)} rows to {args.out}")

if __name__ == "__main__":
    main()
