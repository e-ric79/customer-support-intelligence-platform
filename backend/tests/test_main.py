import pytest
from fastapi.testclient import TestClient
from app.main import app


@pytest.fixture
def client():
    # Context manager form triggers the lifespan startup (model load)
    with TestClient(app) as c:
        yield c


def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_predict_valid_input(client):
    resp = client.post("/predict", json={"text": "I was charged twice for my subscription"})
    assert resp.status_code == 200
    body = resp.json()
    assert "label" in body
    assert 0.0 <= body["confidence"] <= 1.0
    assert isinstance(body["low_confidence"], bool)


def test_predict_empty_string_rejected(client):
    resp = client.post("/predict", json={"text": ""})
    assert resp.status_code == 422  # pydantic min_length=1 validation


def test_predict_malformed_json(client):
    resp = client.post("/predict", json={"wrong_field": "abc"})
    assert resp.status_code == 422
