"""Prompts for Claude code generation — kept minimal for low token usage."""

SYSTEM_PROMPT = """\
You are a healthcare data engineer. Generate ONLY Python code — no explanation.

Rules:
- Define a function: def process(records): ... that takes a list of dicts
- Call it at the end: process(records)
- Use ONLY basic types: dict, list, str, int, float, bool
- NO imports, NO classes, NO print statements
- Keep code under 25 lines
- The variable `records` is provided as input
"""

STEP_PROMPTS = {
    "validate": (
        "Write `process(records)` that validates patient records.\n"
        "Required fields: patient_id, name, dob, ssn, diagnosis_code, lab_results.\n"
        "Return dict: {valid: list, invalid: list[{patient_id, missing}], valid_count: int, invalid_count: int}"
    ),
    "normalize": (
        "Write `process(records)` that normalizes patient record formats.\n"
        "For each record: convert dob from 'YYYY-MM-DD' to 'MM/DD/YYYY',\n"
        "uppercase diagnosis_code, round all float values in lab_results to 1 decimal.\n"
        "Return dict: {records: list, transformations: list[str]} where transformations lists what was normalized."
    ),
    "deidentify": (
        "Write `process(records)` that de-identifies PHI in patient records.\n"
        "Mask: name→'REDACTED', ssn→'***-**-'+last4, address→'REDACTED', dob→'REDACTED'.\n"
        "Preserve all other fields. Return dict: {records: list, fields_masked: list[str]}"
    ),
    "enrich": (
        "Write `process(records)` that enriches patient records with derived fields.\n"
        "For each record add: glucose_flag ('normal' if glucose_mg_dl<100, 'high' if <200, 'critical' if >=200),\n"
        "cholesterol_flag ('normal' if cholesterol_mg_dl<200, 'borderline' if <240, 'high' if >=240),\n"
        "a1c_flag ('normal' if hemoglobin_a1c<5.7, 'prediabetic' if <6.5, 'diabetic' if >=6.5).\n"
        "Read values from lab_results dict. Return dict: {records: list, flags_added: list[str]}"
    ),
    "risk_score": (
        "Write `process(records)` that computes a risk score (0-100) for each patient.\n"
        "Scoring: start at 0. Add 20 if glucose_flag is 'high' or 'critical', add 30 if glucose_flag is 'critical',\n"
        "add 15 if cholesterol_flag is 'high' or 'borderline', add 20 if a1c_flag is 'diabetic',\n"
        "add 10 if a1c_flag is 'prediabetic'. Cap at 100.\n"
        "Add 'risk_score' and 'risk_level' ('low' <30, 'moderate' <60, 'high' >=60) to each record.\n"
        "Return dict: {records: list, high_risk_count: int, avg_risk_score: float (rounded to 1)}"
    ),
    "aggregate": (
        "Write `process(records)` that computes summary statistics.\n"
        "Count records per diagnosis_code, compute avg glucose_mg_dl from lab_results,\n"
        "count records per risk_level, find the highest risk_score.\n"
        "Return dict: {total_records: int, diagnosis_counts: dict, avg_glucose: float (rounded to 1),\n"
        "risk_distribution: dict, max_risk_score: int}"
    ),
}
