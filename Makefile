# Cibles auto-documentées : `make help`
.DEFAULT_GOAL := help

help: ## Affiche cette aide
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-14s\033[0m %s\n", $$1, $$2}'

setup: ## Environnement complet (uv sync)
	uv sync

data: ## MASSIVE fr-FR : téléchargement + contrôles qualité + décontamination
	uv run intents-data

baseline: ## Baseline TF-IDF + régression logistique (la barre à battre)
	uv run intents-train-baseline

train: ## Fine-tuning CamemBERT (GPU conseillé — cf. notebooks/colab.md)
	uv run intents-train-transformer --epochs 8 --lr 5e-5

evaluate: ## Comparaison finale baseline vs CamemBERT (descelle le TEST)
	uv run intents-evaluate

optimize: ## Export ONNX + quantification INT8 per-channel + benchmark
	uv run intents-optimize

serve: ## API de scoring (moteur ONNX INT8) -> http://localhost:8000/docs
	uv run intents-serve

test: ## Suite pytest
	uv run pytest -q

format: ## Formate + autofixe (ruff)
	uv run ruff check --fix . && uv run ruff format .

check: ## Tout ce que la CI vérifie : lint + format + tests
	uv run ruff check . && uv run ruff format --check . && uv run pytest -q

mlflow: ## UI MLflow (runs baseline / transformer / optimisation)
	uv run mlflow ui --backend-store-uri sqlite:///mlflow.db

docker-build: ## Image de serving (sans torch) — nécessite models/onnx (make optimize)
	docker build -t intent-classifier-api .

docker-run: ## API conteneurisée, liée à la boucle locale uniquement
	docker run --rm -p 127.0.0.1:8000:8000 intent-classifier-api
