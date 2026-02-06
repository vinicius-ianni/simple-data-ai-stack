"""TDD tests: AI agent code generation and sandbox execution."""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pydantic_monty
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from healthcare_agent.data import FAKE_PATIENTS


class TestExecuteInSandbox:
    """Test sandbox execution with realistic healthcare code."""

    def test_validate_code_in_sandbox(self):
        """Validation code separates valid/invalid records."""
        from healthcare_agent.agent import execute_in_sandbox

        code = """
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
        result = execute_in_sandbox(code, FAKE_PATIENTS)
        assert result["valid_count"] == 4
        assert result["invalid_count"] == 1

    def test_deidentify_code_in_sandbox(self):
        """De-identification code masks PHI fields."""
        from healthcare_agent.agent import execute_in_sandbox

        code = """
def process(records):
    output = []
    for r in records:
        masked = {}
        for k, v in r.items():
            masked[k] = v
        masked["name"] = "REDACTED"
        masked["ssn"] = "***-**-" + r["ssn"][-4:]
        masked["address"] = "REDACTED"
        output.append(masked)
    return {"records": output, "fields_masked": ["name", "ssn", "address"]}

process(records)
"""
        patients = [p for p in FAKE_PATIENTS if "name" in p]
        result = execute_in_sandbox(code, patients)
        assert result["records"][0]["name"] == "REDACTED"
        assert result["records"][0]["ssn"] == "***-**-6789"
        assert result["records"][0]["address"] == "REDACTED"
        assert result["records"][0]["patient_id"] == "P-1001"  # ID preserved

    def test_aggregate_code_in_sandbox(self):
        """Aggregation code computes summary statistics."""
        from healthcare_agent.agent import execute_in_sandbox

        code = """
def process(records):
    diag_counts = {}
    glucose_values = []
    for r in records:
        code = r["diagnosis_code"]
        diag_counts[code] = diag_counts.get(code, 0) + 1
        glucose = r.get("lab_results", {}).get("glucose_mg_dl")
        if glucose is not None:
            glucose_values.append(glucose)
    avg_glucose = sum(glucose_values) / len(glucose_values) if glucose_values else 0.0
    return {"total_records": len(records), "diagnosis_counts": diag_counts, "avg_glucose": avg_glucose}

process(records)
"""
        patients = [p for p in FAKE_PATIENTS if "name" in p]
        result = execute_in_sandbox(code, patients)
        assert result["total_records"] == 4
        assert result["diagnosis_counts"]["E11.9"] == 2
        assert result["avg_glucose"] > 0

    def test_normalize_code_in_sandbox(self):
        """Normalization code standardizes date formats and rounds lab values."""
        from healthcare_agent.agent import execute_in_sandbox

        code = """
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
        patients = [p for p in FAKE_PATIENTS if "name" in p]
        result = execute_in_sandbox(code, patients)
        assert result["records"][0]["dob"] == "03/15/1985"
        assert len(result["transformations"]) == 3

    def test_enrich_code_in_sandbox(self):
        """Enrichment code adds clinical flags."""
        from healthcare_agent.agent import execute_in_sandbox

        code = """
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
        patients = [p for p in FAKE_PATIENTS if "name" in p]
        result = execute_in_sandbox(code, patients)
        # P-1001: glucose 142 → high, cholesterol 210 → borderline, a1c 7.2 → diabetic
        assert result["records"][0]["glucose_flag"] == "high"
        assert result["records"][0]["cholesterol_flag"] == "borderline"
        assert result["records"][0]["a1c_flag"] == "diabetic"

    def test_risk_score_code_in_sandbox(self):
        """Risk scoring code computes per-patient scores."""
        from healthcare_agent.agent import execute_in_sandbox

        # Pre-enriched records
        code = """
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
        enriched = [
            {"patient_id": "P-1001", "glucose_flag": "high", "cholesterol_flag": "borderline", "a1c_flag": "diabetic"},
            {"patient_id": "P-1002", "glucose_flag": "normal", "cholesterol_flag": "borderline", "a1c_flag": "normal"},
        ]
        result = execute_in_sandbox(code, enriched)
        # P-1001: 20 (glucose high) + 15 (cholesterol borderline) + 20 (a1c diabetic) = 55 → moderate
        assert result["records"][0]["risk_score"] == 55
        assert result["records"][0]["risk_level"] == "moderate"
        assert result["avg_risk_score"] > 0

    def test_sandbox_rejects_malicious_code(self):
        """Agent must handle sandbox errors gracefully."""
        from healthcare_agent.agent import execute_in_sandbox

        with pytest.raises(pydantic_monty.MontyRuntimeError):
            execute_in_sandbox('open("/etc/passwd").read()', [])


class TestGenerateCode:
    """Test Claude API integration for code generation."""

    @patch("healthcare_agent.agent.anthropic")
    def test_generate_code_returns_clean_python(self, mock_anthropic):
        from healthcare_agent.agent import generate_code

        mock_client = MagicMock()
        mock_anthropic.Anthropic.return_value = mock_client
        mock_client.messages.create.return_value = MagicMock(
            content=[MagicMock(text='```python\ndef process(records):\n    return {"count": len(records)}\n\nprocess(records)\n```')]
        )

        code = generate_code("validate", "Validate patient records")
        assert "def process(records)" in code
        assert "```" not in code  # markdown fences stripped

    @patch("healthcare_agent.agent.anthropic")
    def test_generate_code_calls_configured_model(self, mock_anthropic):
        from healthcare_agent.agent import generate_code

        mock_client = MagicMock()
        mock_anthropic.Anthropic.return_value = mock_client
        mock_client.messages.create.return_value = MagicMock(
            content=[MagicMock(text="def process(records):\n    return {}")]
        )

        generate_code("validate", "Validate records")
        call_kwargs = mock_client.messages.create.call_args[1]
        assert "model" in call_kwargs
