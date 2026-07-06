"""Évaluation finale comparée : baseline vs CamemBERT — le TEST est descellé ici.

Discipline : aucune décision de modèle n'a été prise sur le test. Toute
l'itération (epochs, lr) s'est faite sur la validation. Ce module CHARGE les
deux modèles finaux (aucun ré-entraînement) et lit le test — une lecture datée,
tracée dans MLflow. Si un candidat supplémentaire devait être entraîné ensuite,
sa sélection se ferait encore sur la validation, et toute relecture du test
serait annoncée comme telle.

Sorties : tableau comparatif, pires classes, confusions les plus fréquentes,
reports/test_comparison.json + reports/per_class_f1.csv.
"""

from __future__ import annotations

import json
import time

import joblib
import pandas as pd
import structlog
from sklearn.metrics import accuracy_score, classification_report, f1_score

from intent_classifier.common.config import settings
from intent_classifier.modeling.baseline import load_split

log = structlog.get_logger()


# ---------- Prédiction (chargement seul, jamais d'entraînement) ----------
def predict_baseline(texts: list[str]) -> list[str]:
    path = settings.models_dir / "baseline.joblib"
    if not path.exists():
        raise FileNotFoundError(f"{path} absent — lance d'abord `intents-train-baseline`.")
    return list(joblib.load(path).predict(texts))


def predict_transformer(texts: list[str], batch_size: int = 64) -> list[str]:
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    model_dir = settings.models_dir / "camembert"
    if not (model_dir / "model.safetensors").exists():
        raise FileNotFoundError(f"{model_dir} absent — entraîne (ou rapatrie) le modèle d'abord.")

    device = (
        "cuda"
        if torch.cuda.is_available()
        else ("mps" if torch.backends.mps.is_available() else "cpu")
    )
    tok = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(device).eval()
    id2label = model.config.id2label

    preds: list[str] = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = tok(
                texts[i : i + batch_size],
                truncation=True,
                max_length=64,
                padding=True,
                return_tensors="pt",
            ).to(device)
            logits = model(**batch).logits
            preds.extend(id2label[int(j)] for j in logits.argmax(-1).cpu().tolist())
    return preds


# ---------- Analyses (pures, testables sans modèle) ----------
def metric_table(y_true: list[str], preds: dict[str, list[str]]) -> pd.DataFrame:
    rows = {
        name: {
            "macro_f1": f1_score(y_true, y_pred, average="macro"),
            "weighted_f1": f1_score(y_true, y_pred, average="weighted"),
            "accuracy": accuracy_score(y_true, y_pred),
        }
        for name, y_pred in preds.items()
    }
    return pd.DataFrame(rows).T.round(4)


def worst_classes(y_true: list[str], y_pred: list[str], k: int = 10) -> pd.DataFrame:
    """Les k classes les moins bien reconnues (F1 croissant) avec leur support."""
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    rows = [
        {"intent": name, "f1": vals["f1-score"], "support": int(vals["support"])}
        for name, vals in report.items()
        if isinstance(vals, dict)
        and "f1-score" in vals
        and name not in ("macro avg", "weighted avg")
    ]
    return pd.DataFrame(rows).sort_values("f1").head(k).reset_index(drop=True)


def top_confusions(y_true: list[str], y_pred: list[str], k: int = 10) -> pd.DataFrame:
    """Les k paires (vraie -> prédite) les plus confondues."""
    ct = pd.crosstab(pd.Series(y_true, name="vraie"), pd.Series(y_pred, name="prédite"))
    stacked = ct.stack()
    stacked = stacked[
        stacked.index.get_level_values(0) != stacked.index.get_level_values(1)
    ]
    top = stacked.sort_values(ascending=False).head(k)
    return top.rename("n").reset_index()


def main() -> None:
    import mlflow

    test = load_split("test")
    y_true = test["intent"].tolist()
    texts = test["text"].tolist()

    print(f"TEST descellé : {len(test)} exemples · {test['intent'].nunique()} intentions\n")

    t0 = time.perf_counter()
    preds = {"baseline_tfidf": predict_baseline(texts)}
    t_base = time.perf_counter() - t0
    t0 = time.perf_counter()
    preds["camembert_ft"] = predict_transformer(texts)
    t_bert = time.perf_counter() - t0

    table = metric_table(y_true, preds)
    delta = table.loc["camembert_ft", "macro_f1"] - table.loc["baseline_tfidf", "macro_f1"]

    print("=== Comparaison finale (TEST) ===")
    print(table.to_string())
    print(f"\nΔ macro-F1 (camembert - baseline) : {delta:+.4f}")
    print(f"Inférence totale : baseline {t_base:.1f}s · camembert {t_bert:.1f}s")

    worst = worst_classes(y_true, preds["camembert_ft"])
    print("\n=== CamemBERT — 10 pires classes (F1, support) ===")
    print(worst.to_string(index=False))

    confusions = top_confusions(y_true, preds["camembert_ft"])
    print("\n=== CamemBERT — confusions les plus fréquentes ===")
    print(confusions.to_string(index=False))

    settings.reports_dir.mkdir(parents=True, exist_ok=True)
    (settings.reports_dir / "test_comparison.json").write_text(
        json.dumps(table.to_dict(orient="index"), indent=2)
    )
    report = classification_report(
        y_true, preds["camembert_ft"], output_dict=True, zero_division=0
    )
    pd.DataFrame(report).T.to_csv(settings.reports_dir / "per_class_f1.csv")

    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment(settings.experiment_name)
    with mlflow.start_run(run_name="test-comparison"):
        for name in table.index:
            mlflow.log_metrics(
                {f"test_{name}_{m}": float(table.loc[name, m]) for m in table.columns}
            )

    print(f"\nRapports -> {settings.reports_dir}")
    log.info("evaluation_done", delta_macro_f1=round(float(delta), 4))


if __name__ == "__main__":
    main()
