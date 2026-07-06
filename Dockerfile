# Image de serving : API FastAPI + moteur ONNX INT8.
# Volontairement SANS PyTorch (le moteur de prod est onnxruntime) : image ~10x
# plus légère qu'une image d'entraînement.
FROM python:3.12-slim

WORKDIR /app

# 1) Dépendances (couche cachée tant que requirements-serving.txt ne change pas)
COPY requirements-serving.txt .
RUN pip install --no-cache-dir -r requirements-serving.txt

# 2) Code + modèle quantifié (le fp32 et torch sont exclus via .dockerignore)
COPY src/ src/
COPY models/onnx/ models/onnx/

ENV PYTHONPATH=/app/src
EXPOSE 8000

CMD ["uvicorn", "intent_classifier.serving.api:app", "--host", "0.0.0.0", "--port", "8000"]
