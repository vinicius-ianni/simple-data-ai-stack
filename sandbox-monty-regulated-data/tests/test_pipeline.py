"""TDD tests: End-to-end pipeline orchestration."""

import sys
from pathlib import Path
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Realistic code that Claude would generate for each step

MOCK_VALIDATE_CODE = """
def process(records):
    required = ["patient_id", "name", "dob", "ssn", "diagnosis_code", "lab_results"]
    valid = []
    invalid = []
    for r in records:
        missing = [f for f in required if f not in r]
        if missing:
            invalid.append({"patient_id": r.get("patient_id", "unknown"), "missing": missing})
        else:
            valid.append(r)
    return {"valid": valid, "invalid": invalid, "valid_count": len(valid), "invalid_count": len(invalid)}

process(records)
"""

MOCK_NORMALIZE_CODE = """
def process(records):
    output = []
    for r in records:
        rec = {}
        for k, v in r.items():
            rec[k] = v
        parts = rec["dob"].split("-")
        rec["dob"] = parts[1] + "/" + parts[2] + "/" + parts[0]
        rec["diagnosis_code"] = rec["diagnosis_code"].upper()
        labs = {}
        for lk, lv in rec.get("lab_results", {}).items():
            labs[lk] = round(lv, 1) if isinstance(lv, float) else lv
        rec["lab_results"] = labs
        output.append(rec)
    return {"records": output, "transformations": ["dob_format", "diagnosis_uppercase", "lab_rounding"]}

process(records)
"""

MOCK_DEIDENTIFY_CODE = """
def process(records):
    output = []
    for r in records:
        masked = {}
        for k, v in r.items():
            masked[k] = v
        masked["name"] = "REDACTED"
        masked["ssn"] = "***-**-" + r["ssn"][-4:]
        masked["address"] = "REDACTED"
        masked["dob"] = "REDACTED"
        output.append(masked)
    return {"records": output, "fields_masked": ["name", "ssn", "address", "dob"]}

process(records)
"""

MOCK_ENRICH_CODE = """
def process(records):
    output = []
    for r in records:
        rec = {}
        for k, v in r.items():
            rec[k] = v
        labs = rec.get("lab_results", {})
        g = labs.get("glucose_mg_dl", 0)
        rec["glucose_flag"] = "normal" if g < 100 else ("high" if g < 200 else "critical")
        c = labs.get("cholesterol_mg_dl", 0)
        rec["cholesterol_flag"] = "normal" if c < 200 else ("borderline" if c < 240 else "high")
        a = labs.get("hemoglobin_a1c", 0)
        rec["a1c_flag"] = "normal" if a < 5.7 else ("prediabetic" if a < 6.5 else "diabetic")
        output.append(rec)
    return {"records": output, "flags_added": ["glucose_flag", "cholesterol_flag", "a1c_flag"]}

process(records)
"""

MOCK_RISK_SCORE_CODE = """
def process(records):
    output = []
    high_risk = 0
    total_score = 0
    for r in records:
        rec = {}
        for k, v in r.items():
            rec[k] = v
        score = 0
        if rec.get("glucose_flag") == "high":
            score = score + 20
        if rec.get("glucose_flag") == "critical":
            score = score + 50
        if rec.get("cholesterol_flag") in ["high", "borderline"]:
            score = score + 15
        if rec.get("a1c_flag") == "diabetic":
            score = score + 20
        if rec.get("a1c_flag") == "prediabetic":
            score = score + 10
        if score > 100:
            score = 100
        rec["risk_score"] = score
        rec["risk_level"] = "low" if score < 30 else ("moderate" if score < 60 else "high")
        total_score = total_score + score
        if rec["risk_level"] == "high":
            high_risk = high_risk + 1
        output.append(rec)
    avg = round(total_score / len(output), 1) if output else 0.0
    return {"records": output, "high_risk_count": high_risk, "avg_risk_score": avg}

process(records)
"""

MOCK_AGGREGATE_CODE = """
def process(records):
    diag_counts = {}
    glucose_values = []
    risk_dist = {}
    max_risk = 0
    for r in records:
        code = r["diagnosis_code"]
        diag_counts[code] = diag_counts.get(code, 0) + 1
        glucose = r.get("lab_results", {}).get("glucose_mg_dl")
        if glucose is not None:
            glucose_values.append(glucose)
        rl = r.get("risk_level", "unknown")
        risk_dist[rl] = risk_dist.get(rl, 0) + 1
        rs = r.get("risk_score", 0)
        if rs > max_risk:
            max_risk = rs
    avg_glucose = sum(glucose_values) / len(glucose_values) if glucose_values else 0.0
    return {"total_records": len(records), "diagnosis_counts": diag_counts, "avg_glucose": round(avg_glucose, 1), "risk_distribution": risk_dist, "max_risk_score": max_risk}

process(records)
"""

MOCK_CODE_MAP = {
    "validate": MOCK_VALIDATE_CODE,
    "normalize": MOCK_NORMALIZE_CODE,
    "deidentify": MOCK_DEIDENTIFY_CODE,
    "enrich": MOCK_ENRICH_CODE,
    "risk_score": MOCK_RISK_SCORE_CODE,
    "aggregate": MOCK_AGGREGATE_CODE,
}


def _mock_generate(step_name, prompt):
    """Return realistic code per step — simulates Claude."""
    return MOCK_CODE_MAP[step_name]


class TestPipelineEndToEnd:
    """Full pipeline test with mocked Claude API."""

    @patch("healthcare_agent.pipeline.generate_code")
    def test_full_pipeline_produces_results(self, mock_gen):
        from healthcare_agent.pipeline import run_pipeline_headless

        mock_gen.side_effect = _mock_generate
        results = run_pipeline_headless()

        # Step 1: validation
        assert results["validate"]["valid_count"] == 4
        assert results["validate"]["invalid_count"] == 1

        # Step 2: normalize
        assert "records" in results["normalize"]
        assert "dob_format" in results["normalize"]["transformations"]

        # Step 3: de-identification
        deidentified = results["deidentify"]["records"]
        assert all(r["name"] == "REDACTED" for r in deidentified)
        assert all(r["ssn"].startswith("***-**-") for r in deidentified)

        # Step 4: enrich
        enriched = results["enrich"]["records"]
        assert all("glucose_flag" in r for r in enriched)
        assert all("cholesterol_flag" in r for r in enriched)
        assert all("a1c_flag" in r for r in enriched)

        # Step 5: risk scores
        assert "high_risk_count" in results["risk_score"]
        assert "avg_risk_score" in results["risk_score"]
        scored = results["risk_score"]["records"]
        assert all("risk_score" in r for r in scored)
        assert all("risk_level" in r for r in scored)

        # Step 6: aggregation
        assert results["aggregate"]["total_records"] == 4
        assert "E11.9" in results["aggregate"]["diagnosis_counts"]
        assert "risk_distribution" in results["aggregate"]
        assert "max_risk_score" in results["aggregate"]

    @patch("healthcare_agent.pipeline.generate_code")
    def test_pipeline_chains_data_correctly(self, mock_gen):
        """Output of step N feeds into step N+1."""
        from healthcare_agent.pipeline import run_pipeline_headless

        mock_gen.side_effect = _mock_generate
        results = run_pipeline_headless()

        # All steps after validate should only see 4 valid records
        assert results["aggregate"]["total_records"] == 4
        assert len(results["risk_score"]["records"]) == 4
        assert len(results["enrich"]["records"]) == 4
