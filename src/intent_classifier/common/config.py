"""Configuration centralisée (pydantic-settings).

Une instance unique `settings` : chemins, graine, dataset, modèles. Typée,
surchargée par variables d'environnement / .env, graine partagée partout —
la même discipline que sur readmission-risk-ml.
"""

from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

# Racine du projet (…/french-intent-classifier), calculée depuis ce fichier.
PROJECT_ROOT = Path(__file__).resolve().parents[3]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # --- Reproductibilité ---
    random_seed: int = 42

    # --- Chemins ---
    data_dir: Path = PROJECT_ROOT / "data"
    models_dir: Path = PROJECT_ROOT / "models"
    reports_dir: Path = PROJECT_ROOT / "reports"

    # --- Données : MASSIVE (Amazon Science), locale française ---
    # ~11 500 phrases d'entraînement, 60 intentions réelles (assistants vocaux).
    # On utilise le miroir SetFit « données pures » : le repo officiel
    # AmazonScience/massive passe par un script de chargement (massive.py),
    # que datasets >= 3 ne supporte plus.
    dataset_name: str = "SetFit/amazon_massive_intent_fr-FR"
    text_col: str = "text"
    label_col: str = "label"
    intent_col: str = "label_text"  # nom lisible de l'intention (ex. alarm_set)

    # --- Baseline (étape 3) ---
    baseline_max_features: int = 30_000  # vocabulaire TF-IDF

    # --- Transformer (étape 4) ---
    transformer_model: str = "camembert-base"

    # --- Suivi d'expériences ---
    mlflow_tracking_uri: str = "sqlite:///mlflow.db"
    experiment_name: str = "intents-fr"


settings = Settings()
