"""Fake healthcare patient records for pipeline demo."""

FAKE_PATIENTS = [
    {
        "patient_id": "P-1001",
        "name": "John Smith",
        "dob": "1985-03-15",
        "ssn": "123-45-6789",
        "address": "742 Evergreen Terrace, Springfield, IL 62704",
        "diagnosis_code": "E11.9",
        "diagnosis_desc": "Type 2 diabetes mellitus without complications",
        "lab_results": {
            "glucose_mg_dl": 142.0,
            "hemoglobin_a1c": 7.2,
            "creatinine_mg_dl": 0.9,
            "cholesterol_mg_dl": 210.0,
        },
    },
    {
        "patient_id": "P-1002",
        "name": "Maria Garcia",
        "dob": "1972-11-08",
        "ssn": "987-65-4321",
        "address": "123 Oak Street, Portland, OR 97201",
        "diagnosis_code": "I10",
        "diagnosis_desc": "Essential (primary) hypertension",
        "lab_results": {
            "glucose_mg_dl": 98.0,
            "hemoglobin_a1c": 5.4,
            "creatinine_mg_dl": 1.1,
            "cholesterol_mg_dl": 245.0,
        },
    },
    {
        "patient_id": "P-1003",
        "name": "Robert Johnson",
        "dob": "1990-07-22",
        "ssn": "456-78-9012",
        "address": "456 Pine Ave, Austin, TX 78701",
        "diagnosis_code": "E11.9",
        "diagnosis_desc": "Type 2 diabetes mellitus without complications",
        "lab_results": {
            "glucose_mg_dl": 165.0,
            "hemoglobin_a1c": 8.1,
            "creatinine_mg_dl": 1.3,
            "cholesterol_mg_dl": 198.0,
        },
    },
    {
        "patient_id": "P-1004",
        "name": "Sarah Chen",
        "dob": "1968-01-30",
        "ssn": "321-54-9876",
        "address": "789 Maple Dr, Seattle, WA 98101",
        "diagnosis_code": "J45.20",
        "diagnosis_desc": "Mild intermittent asthma, uncomplicated",
        "lab_results": {
            "glucose_mg_dl": 92.0,
            "hemoglobin_a1c": 5.1,
            "creatinine_mg_dl": 0.8,
            "cholesterol_mg_dl": 180.0,
        },
    },
    {
        # Intentionally invalid record — missing name, ssn, address
        "patient_id": "P-1005",
        "dob": "1995-12-01",
        "diagnosis_code": "R51",
        "diagnosis_desc": "Headache",
        "lab_results": {
            "glucose_mg_dl": 105.0,
        },
    },
]

REQUIRED_FIELDS = [
    "patient_id",
    "name",
    "dob",
    "ssn",
    "address",
    "diagnosis_code",
    "diagnosis_desc",
    "lab_results",
]
