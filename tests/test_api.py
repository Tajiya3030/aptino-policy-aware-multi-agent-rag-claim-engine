import pytest
from fastapi.testclient import TestClient
from backend.main import app

client = TestClient(app)

def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "vector_store_status" in data

def test_analyze_endpoint():
    sample_case = {
        "case_id": "TEST-API-001",
        "sum_insured_inr": 500000,
        "continuous_coverage_months": 24,
        "patient": {"age": 30},
        "hospital": {"network_provider": True},
        "treatment": {"type": "inpatient", "admission_hours": 48, "diagnosis": "Viral fever", "procedure": "Medical management"},
        "expenses_inr": {"room": 10000, "doctor_fees": 15000, "medicines_diagnostics": 20000},
        "documents": ["claim_form", "discharge_summary", "itemized_bill"]
    }
    response = client.post("/analyze", json=sample_case)
    assert response.status_code == 200
    data = response.json()
    assert data["case_id"] == "TEST-API-001"
    assert "decision" in data
    assert "retrieval_metadata" in data
    assert "validation" in data
