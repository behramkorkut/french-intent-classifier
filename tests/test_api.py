"""Tests de l'API : post-traitement pur + intégration (skip si modèle absent)."""

import numpy as np
import pytest
from fastapi.testclient import TestClient

from intent_classifier.common.config import settings
from intent_classifier.serving.api import app, softmax, top_intents

needs_model = pytest.mark.skipif(
    not (settings.models_dir / "onnx" / "model_quantized.onnx").exists(),
    reason="Modèle ONNX INT8 absent — lance `intents-optimize` d'abord.",
)


# ---------- Pur ----------
def test_softmax_sums_to_one_and_is_stable():
    probs = softmax(np.array([1000.0, 1001.0, 999.0]))  # grands logits : pas d'overflow
    assert probs.sum() == pytest.approx(1.0)
    assert probs.argmax() == 1


def test_top_intents_ordered_and_labelled():
    id2label = {0: "alarm_set", 1: "play_music", 2: "weather_query"}
    top = top_intents(np.array([0.1, 3.0, 1.0]), id2label, k=2)
    assert [t["intent"] for t in top] == ["play_music", "weather_query"]
    assert top[0]["confidence"] > top[1]["confidence"]
    assert all(0.0 < t["confidence"] <= 1.0 for t in top)


# ---------- Intégration (modèle réel) ----------
def test_health_is_always_up():
    body = TestClient(app).get("/health").json()
    assert body["status"] == "ok"
    assert "engine" in body


@needs_model
def test_predict_returns_intent_confidence_latency():
    resp = TestClient(app).post("/predict", json={"text": "mets une alarme à sept heures"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["intent"] == "alarm_set"
    assert 0.0 < body["confidence"] <= 1.0
    assert len(body["top"]) == 3
    assert body["latency_ms"] > 0


@needs_model
def test_predict_rejects_invalid_payload():
    client = TestClient(app)
    assert client.post("/predict", json={"text": ""}).status_code == 422
    assert client.post("/predict", json={}).status_code == 422
