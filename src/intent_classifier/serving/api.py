"""API de scoring : classification d'intentions en français.

Le moteur de production est l'**ONNX INT8 per-channel** — choisi par mesure
(étape 6) : x6,5 vs PyTorch CPU pour -0,3 point de macro-F1, 112 Mo. L'image
Docker de serving n'embarque ni PyTorch ni le modèle fp32.

Endpoints :
- GET  /health  : état du service (modèle chargé ?)
- POST /predict : texte -> intention + confiance + top-3 + latence

Le moteur est injecté via une dépendance FastAPI (surchargeable en test),
chargé une seule fois par process.
"""

from __future__ import annotations

import time
from functools import lru_cache
from typing import Annotated

import numpy as np
import uvicorn
from fastapi import Depends, FastAPI, HTTPException
from pydantic import BaseModel, Field

from intent_classifier.common.config import settings

ENGINE_NAME = "onnxruntime-int8-per-channel"

app = FastAPI(
    title="French Intent Classifier API",
    version="1.0",
    description=(
        "Classification d'intentions en français (60 classes, MASSIVE fr-FR) — "
        "CamemBERT fine-tuné, servi en ONNX INT8 quantifié par canal."
    ),
)


# ---------- Post-traitement (pur, testable sans modèle) ----------
def softmax(logits: np.ndarray) -> np.ndarray:
    z = np.asarray(logits, dtype=float).ravel()
    z = z - z.max()  # stabilité numérique
    e = np.exp(z)
    return e / e.sum()


def top_intents(logits: np.ndarray, id2label: dict[int, str], k: int = 3) -> list[dict]:
    probs = softmax(logits)
    order = np.argsort(probs)[::-1][:k]
    return [
        {"intent": id2label[int(i)], "confidence": round(float(probs[i]), 4)} for i in order
    ]


# ---------- Moteur d'inférence ----------
class Engine:
    def __init__(self) -> None:
        from optimum.onnxruntime import ORTModelForSequenceClassification
        from transformers import AutoTokenizer

        onnx_dir = settings.models_dir / "onnx"
        if not (onnx_dir / "model_quantized.onnx").exists():
            raise RuntimeError(
                f"Modèle absent : {onnx_dir}/model_quantized.onnx — lance `intents-optimize`."
            )
        self.tokenizer = AutoTokenizer.from_pretrained(onnx_dir)
        self.model = ORTModelForSequenceClassification.from_pretrained(
            onnx_dir, file_name="model_quantized.onnx"
        )
        self.id2label = {int(i): name for i, name in self.model.config.id2label.items()}

    def logits(self, text: str) -> np.ndarray:
        enc = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=64)
        out = self.model(**enc).logits
        return out.detach().numpy() if hasattr(out, "detach") else np.asarray(out)


@lru_cache(maxsize=1)
def _engine_singleton() -> Engine:
    return Engine()


def get_engine() -> Engine:
    try:
        return _engine_singleton()
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


EngineDep = Annotated[Engine, Depends(get_engine)]


# ---------- Schémas ----------
class PredictRequest(BaseModel):
    text: str = Field(
        ..., min_length=1, max_length=500, examples=["règle une alarme à sept heures demain"]
    )


class Prediction(BaseModel):
    intent: str
    confidence: float


class PredictResponse(BaseModel):
    intent: str
    confidence: float
    top: list[Prediction]
    latency_ms: float
    engine: str


# ---------- Endpoints ----------
@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "model_loaded": (settings.models_dir / "onnx" / "model_quantized.onnx").exists(),
        "engine": ENGINE_NAME,
    }


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest, engine: EngineDep) -> PredictResponse:
    t0 = time.perf_counter()
    top = top_intents(engine.logits(req.text), engine.id2label)
    latency_ms = round((time.perf_counter() - t0) * 1000, 1)
    return PredictResponse(
        intent=top[0]["intent"],
        confidence=top[0]["confidence"],
        top=[Prediction(**t) for t in top],
        latency_ms=latency_ms,
        engine=ENGINE_NAME,
    )


def run() -> None:
    """Point d'entrée `intents-serve`."""
    uvicorn.run("intent_classifier.serving.api:app", host="0.0.0.0", port=8000)


if __name__ == "__main__":
    run()
