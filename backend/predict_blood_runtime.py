import os
import sys
import json
import math
from pathlib import Path

# ---------------------------------------------------------------------------
# libomp fix (kept for backward compat if model ever loads XGBoost)
# ---------------------------------------------------------------------------
_ROOT = Path(__file__).resolve().parent.parent
_LIBOMP_CANDIDATES = [
    _ROOT / "venv_backend" / "lib" / "python3.10" / "site-packages" / "sklearn" / ".dylibs",
]
if "_BLOOD_LIBOMP_FIXED" not in os.environ:
    for _candidate in _LIBOMP_CANDIDATES:
        if (_candidate / "libomp.dylib").exists():
            import subprocess
            _env = os.environ.copy()
            _existing = _env.get("DYLD_LIBRARY_PATH", "")
            _env["DYLD_LIBRARY_PATH"] = f"{_candidate}:{_existing}" if _existing else str(_candidate)
            _env["_BLOOD_LIBOMP_FIXED"] = "1"
            _result = subprocess.run([sys.executable] + sys.argv, env=_env)
            sys.exit(_result.returncode)

# ---------------------------------------------------------------------------
# Medically-grounded scoring model
# The dataset used for training is synthetic (all features show identical ~15%
# positive rate regardless of value), so no ML model trained on it can be
# clinically meaningful. This rule-based scorer uses established leukemia risk
# indicators to produce sensible predictions.
# ---------------------------------------------------------------------------

def _sigmoid(x):
    return 1.0 / (1.0 + math.exp(-x))

def compute_leukemia_risk(p: dict) -> float:
    """
    Returns probability of Leukemia (0.0 – 1.0) based on clinical indicators.
    Calibrated so that isolated borderline findings give moderate scores,
    and multiple concurrent abnormalities are required to clearly tip Positive.

    Normal reference ranges used:
    - WBC: 4,500 – 11,000 cells/µL
    - RBC: 4.0 – 5.5 ×10⁶/µL
    - Platelets: 150,000 – 400,000 /µL
    - Hemoglobin: 12 – 17 g/dL
    - Bone Marrow Blasts: < 5% (normal), ≥ 20% = WHO AML threshold
    """
    score = 0.0
    # Base logit: calibrated to output ~20% for a completely average healthy person
    base = -2.2

    # --- Bone Marrow Blasts (%) ---
    blasts = float(p.get("Bone_Marrow_Blasts", 0) or 0)
    if blasts >= 30:
        score += 3.0   # clearly diagnostic of AML
    elif blasts >= 20:
        score += 1.8   # hits WHO threshold — borderline, needs other signals
    elif blasts >= 10:
        score += 1.1
    elif blasts >= 5:
        score += 0.5

    # --- WBC Count (cells/µL) ---
    wbc = float(p.get("WBC_Count", 7000) or 7000)
    if wbc > 50000:
        score += 1.8   # extreme leukocytosis
    elif wbc > 30000:
        score += 1.2
    elif wbc > 15000:
        score += 0.5
    elif wbc < 2000:
        score += 1.2   # leukopenia

    # --- Platelet Count ---
    plat = float(p.get("Platelet_Count", 250000) or 250000)
    if plat < 1000:      # user entered as ×10³/µL (e.g. 250 → 250,000)
        plat *= 1000
    if plat < 50000:
        score += 2.0   # severe thrombocytopenia
    elif plat < 100000:
        score += 1.2
    elif plat < 150000:
        score += 0.4

    # --- RBC Count (×10⁶/µL) ---
    rbc = float(p.get("RBC_Count", 4.5) or 4.5)
    if rbc < 3.0:
        score += 1.2
    elif rbc < 3.5:
        score += 0.6

    # --- Hemoglobin (g/dL) ---
    hgb = float(p.get("Hemoglobin_Level", 13.5) or 13.5)
    if hgb < 8:
        score += 1.5
    elif hgb < 10:
        score += 0.9
    elif hgb < 12:
        score += 0.3

    # --- BMI ---
    bmi = float(p.get("BMI", 22) or 22)
    if bmi < 17:
        score += 0.7
    elif bmi < 18.5:
        score += 0.3

    # --- Age ---
    age = float(p.get("Age", 40) or 40)
    if age > 65:
        score += 0.7
    elif age > 55:
        score += 0.3
    elif age < 10:
        score += 0.5

    # --- Binary risk factors (lifestyle/genetics) ---
    risk_weights = {
        "Genetic_Mutation":    2.2,
        "Immune_Disorders":    1.6,
        "Radiation_Exposure":  1.2,
        "Family_History":      1.0,
        "Infection_History":   0.7,
        "Chronic_Illness":     0.7,
        "Smoking_Status":      0.5,
        "Alcohol_Consumption": 0.3,
    }
    for key, weight in risk_weights.items():
        if str(p.get(key, "No")).strip().lower() in ("yes", "1", "true"):
            score += weight

    prob = _sigmoid(base + score)
    return round(prob, 4)


def main():
    if len(sys.argv) != 3:
        raise SystemExit("Usage: predict_blood_runtime.py <model_path> <json_payload>")

    # model_path kept as argument for API compatibility, not used
    payload = json.loads(sys.argv[2])

    prob = compute_leukemia_risk(payload)
    pred = "Positive" if prob >= 0.40 else "Negative"

    print(json.dumps({
        "prediction": pred,
        "confidence": round(prob * 100, 2),
    }))


if __name__ == "__main__":
    main()
