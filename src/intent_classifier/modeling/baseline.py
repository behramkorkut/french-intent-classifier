"""Baseline honnête : TF-IDF + régression logistique.

Pourquoi une baseline d'abord : un transformer coûte cher (GPU, latence,
maintenance). Il ne mérite sa place que s'il bat NETTEMENT un modèle linéaire
entraîné en quelques secondes sur CPU. La baseline fixe la barre — et sur de la
classification de phrases courtes, TF-IDF + logistique est étonnamment fort.

Discipline d'évaluation : on itère sur la VALIDATION uniquement. Le TEST reste
sous scellés jusqu'à la comparaison finale baseline vs CamemBERT (étape 5) —
une seule lecture, pas de sélection de modèle dessus.

Métrique principale : macro-F1 (déséquilibre 202x — l'accuracy serait dominée
par calendar_set et ses 810 exemples).
"""

from __future__ import annotations

import time

import joblib
import mlflow
import pandas as pd
import structlog
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import Pipeline

from intent_classifier.common.config import settings

log = structlog.get_logger()

BASELINE_PARAMS = {
    "ngram_range": (1, 2),
    "min_df": 2,
    "sublinear_tf": True,
    "C": 4.0,
    "class_weight": "balanced",  # 202x de déséquilibre : chaque classe compte
}


def load_split(split: str) -> pd.DataFrame:
    path = settings.data_dir / f"{split}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"{path} absent — lance d'abord `intents-data`.")
    return pd.read_parquet(path)


def build_pipeline(min_df: int = BASELINE_PARAMS["min_df"]) -> Pipeline:
    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    ngram_range=BASELINE_PARAMS["ngram_range"],
                    max_features=settings.baseline_max_features,
                    sublinear_tf=BASELINE_PARAMS["sublinear_tf"],
                    min_df=min_df,
                ),
            ),
            (
                "clf",
                LogisticRegression(
                    max_iter=2000,
                    C=BASELINE_PARAMS["C"],
                    class_weight=BASELINE_PARAMS["class_weight"],
                    random_state=settings.random_seed,
                ),
            ),
        ]
    )


def evaluate(model: Pipeline, df: pd.DataFrame) -> dict[str, float]:
    """Macro-F1 (primaire), weighted-F1 et accuracy (lecture)."""
    pred = model.predict(df["text"])
    return {
        "macro_f1": float(f1_score(df["intent"], pred, average="macro")),
        "weighted_f1": float(f1_score(df["intent"], pred, average="weighted")),
        "accuracy": float(accuracy_score(df["intent"], pred)),
    }


def main() -> None:
    train, val = load_split("train"), load_split("validation")

    model = build_pipeline()
    t0 = time.perf_counter()
    model.fit(train["text"], train["intent"])
    fit_s = time.perf_counter() - t0

    metrics = evaluate(model, val)

    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment(settings.experiment_name)
    with mlflow.start_run(run_name="baseline-tfidf-logreg"):
        mlflow.log_params(
            {
                **{k: str(v) for k, v in BASELINE_PARAMS.items()},
                "max_features": settings.baseline_max_features,
                "model": "tfidf+logreg",
                "train_size": len(train),
            }
        )
        mlflow.log_metrics({**{f"val_{k}": v for k, v in metrics.items()}, "fit_seconds": fit_s})

    settings.models_dir.mkdir(parents=True, exist_ok=True)
    out = settings.models_dir / "baseline.joblib"
    joblib.dump(model, out)

    print("\n=== Baseline TF-IDF + logistique (VALIDATION) ===")
    print(f"macro-F1    : {metrics['macro_f1']:.4f}   <- métrique primaire")
    print(f"weighted-F1 : {metrics['weighted_f1']:.4f}")
    print(f"accuracy    : {metrics['accuracy']:.4f}")
    print(f"entraînement: {fit_s:.1f}s (CPU)")
    print(f"\nModèle -> {out}")
    print("Rappel : le TEST reste sous scellés jusqu'à la comparaison finale.")
    log.info("baseline_done", **metrics)


if __name__ == "__main__":
    main()
