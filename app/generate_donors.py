"""Generate synthetic donor health data for local RAG testing."""
import random, datetime as dt
import pandas as pd
from pathlib import Path

DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "donors.csv"

def generate(n: int = 200, seed: int = 42):
    random.seed(seed)
    start_date = dt.date(2024, 1, 1)
    rows = []
    for i in range(n):
        donor_id = f"D{1000+i:04d}"
        sex = random.choice(["M","F"])
        age = int(random.gauss(35, 10))
        age = max(18, min(70, age))
        hb = round(random.gauss(14 if sex=="M" else 13, 1.1), 1)   # g/dL
        sbp = int(random.gauss(122, 14))                           # systolic
        dbp = int(random.gauss(78, 10))                            # diastolic
        bmi = round(random.gauss(24.5, 4.2), 1)
        last_date = start_date + dt.timedelta(days=random.randint(0, 450))
        qflags = random.sample(
            ["recent_travel","recent_antibiotics","tattoo_3m","recent_surgery","none"],
            k=1
        )[0]
        rows.append({
            "donor_id": donor_id,
            "sex": sex,
            "age": age,
            "hb_g_dl": hb,
            "systolic_bp": sbp,
            "diastolic_bp": dbp,
            "bmi": bmi,
            "last_donation_date": last_date.isoformat(),
            "questionnaire_flags": qflags
        })

    df = pd.DataFrame(rows)
    df.to_csv(DATA_PATH, index=False)
    print(f"Generated {n} synthetic donors into {DATA_PATH}")

if __name__ == "__main__":
    generate()
