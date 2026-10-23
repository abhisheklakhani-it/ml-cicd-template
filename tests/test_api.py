from __future__ import annotations

from fastapi.testclient import TestClient

from ml_service.model import load_data


def test_health(client: TestClient) -> None:
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_model_info(client: TestClient) -> None:
    body = client.get("/model-info").json()
    assert len(body["feature_names"]) == 13
    assert body["class_names"] == ["class_0", "class_1", "class_2"]
    assert body["metrics"]["accuracy"] >= 0.9


def test_predict_matches_true_labels(client: TestClient) -> None:
    x, y, _, names = load_data()
    rows = [x[0].tolist(), x[70].tolist(), x[170].tolist()]
    r = client.post("/predict", json={"rows": rows})
    assert r.status_code == 200
    labels = [p["label"] for p in r.json()["predictions"]]
    assert labels == [names[y[0]], names[y[70]], names[y[170]]]


def test_predict_rejects_wrong_feature_count(client: TestClient) -> None:
    r = client.post("/predict", json={"rows": [[1.0, 2.0]]})
    assert r.status_code == 422
    assert "expected 13" in r.json()["detail"]


def test_predict_rejects_empty_and_oversized(client: TestClient) -> None:
    assert client.post("/predict", json={"rows": []}).status_code == 422
    too_many = [[0.0] * 13] * 257
    assert client.post("/predict", json={"rows": too_many}).status_code == 422


def test_predict_rejects_non_numbers(client: TestClient) -> None:
    r = client.post("/predict", json={"rows": [["a"] * 13]})
    assert r.status_code == 422
