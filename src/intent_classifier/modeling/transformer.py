"""Fine-tuning CamemBERT sur les intentions MASSIVE fr-FR.

Philosophie : le notebook Colab n'est qu'un EXÉCUTEUR. Toute la logique vit ici,
dans le paquet — versionnée, testée, exécutable à l'identique sur T4 (CUDA),
Mac (MPS) ou CPU. Sur Colab : `git clone` + `uv run intents-train-transformer`.

Même discipline que la baseline : itération sur la VALIDATION uniquement,
sélection du meilleur epoch sur la macro-F1, le TEST reste sous scellés.

Sorties : models/camembert/ (modèle + tokenizer + metrics.json) + run MLflow.
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np
import pandas as pd
import structlog
from sklearn.metrics import accuracy_score, f1_score

from intent_classifier.common.config import settings
from intent_classifier.modeling.baseline import load_split

log = structlog.get_logger()


def build_label_maps(train: pd.DataFrame) -> tuple[dict[str, int], dict[int, str]]:
    """Mapping intention <-> id, trié pour être déterministe d'un run à l'autre."""
    names = sorted(train["intent"].unique())
    label2id = {name: i for i, name in enumerate(names)}
    return label2id, {i: name for name, i in label2id.items()}


def compute_metrics(eval_pred) -> dict[str, float]:
    """Mêmes métriques que la baseline — comparaison à armes égales."""
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "macro_f1": float(f1_score(labels, preds, average="macro")),
        "weighted_f1": float(f1_score(labels, preds, average="weighted")),
        "accuracy": float(accuracy_score(labels, preds)),
    }


def main() -> None:
    # Imports lourds ici : le module reste importable (et testable) sans GPU.
    import mlflow
    import torch
    from datasets import Dataset
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        Trainer,
        TrainingArguments,
        set_seed,
    )

    parser = argparse.ArgumentParser(description="Fine-tuning CamemBERT (intentions fr)")
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Sous-échantillonne le train (smoke test local, ex. 500)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Autorise l'écrasement d'un modèle existant dans models/camembert",
    )
    args = parser.parse_args()

    # Garde-fou : ne jamais écraser silencieusement un modèle entraîné
    # (ex. le run Colab rapatrié) avec un run local ou un smoke test.
    out_check = settings.models_dir / "camembert" / "model.safetensors"
    if out_check.exists() and not args.overwrite:
        raise SystemExit(
            f"Un modèle existe déjà ({out_check}).\n"
            "Relance avec --overwrite si tu veux vraiment l'écraser."
        )

    set_seed(settings.random_seed)
    train, val = load_split("train"), load_split("validation")

    # Les mappings viennent du train COMPLET (même en smoke test, les 60 classes
    # gardent des ids stables).
    label2id, id2label = build_label_maps(train)
    if args.max_samples:
        train = train.sample(
            n=min(args.max_samples, len(train)), random_state=settings.random_seed
        )
        log.info("smoke_test", train_size=len(train))

    tok = AutoTokenizer.from_pretrained(settings.transformer_model)

    def to_dataset(df: pd.DataFrame) -> Dataset:
        ds = Dataset.from_pandas(df[["text", "intent"]], preserve_index=False)
        ds = ds.map(
            lambda batch: {"labels": [label2id[i] for i in batch["intent"]]}, batched=True
        )
        ds = ds.map(lambda batch: tok(batch["text"], truncation=True, max_length=64), batched=True)
        return ds.remove_columns(["text", "intent"])

    model = AutoModelForSequenceClassification.from_pretrained(
        settings.transformer_model,
        num_labels=len(label2id),
        id2label=id2label,
        label2id=label2id,
    )

    device = (
        "cuda"
        if torch.cuda.is_available()
        else ("mps" if torch.backends.mps.is_available() else "cpu")
    )
    out_dir = settings.models_dir / "camembert"
    targs = TrainingArguments(
        output_dir=str(out_dir / "checkpoints"),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=64,
        learning_rate=args.lr,
        warmup_ratio=0.06,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,          # le meilleur epoch, pas le dernier
        metric_for_best_model="macro_f1",
        fp16=torch.cuda.is_available(),       # mixed precision sur T4
        seed=settings.random_seed,
        logging_steps=50,
        save_total_limit=1,
        report_to=[],
    )
    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=to_dataset(train),
        eval_dataset=to_dataset(val),
        processing_class=tok,
        compute_metrics=compute_metrics,
    )

    t0 = time.perf_counter()
    trainer.train()
    train_s = time.perf_counter() - t0
    metrics = {k.removeprefix("eval_"): v for k, v in trainer.evaluate().items()}

    trainer.save_model(str(out_dir))
    tok.save_pretrained(str(out_dir))
    summary = {
        "model": settings.transformer_model,
        "device": device,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "train_size": len(train),
        "train_seconds": round(train_s, 1),
        "val_macro_f1": metrics["macro_f1"],
        "val_weighted_f1": metrics["weighted_f1"],
        "val_accuracy": metrics["accuracy"],
    }
    (out_dir / "metrics.json").write_text(json.dumps(summary, indent=2))

    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment(settings.experiment_name)
    with mlflow.start_run(run_name=f"camembert-ft-{device}"):
        mlflow.log_params(
            {k: summary[k] for k in ("model", "device", "epochs", "batch_size", "lr", "train_size")}
        )
        mlflow.log_metrics(
            {
                "val_macro_f1": summary["val_macro_f1"],
                "val_weighted_f1": summary["val_weighted_f1"],
                "val_accuracy": summary["val_accuracy"],
                "train_seconds": summary["train_seconds"],
            }
        )

    print("\n=== CamemBERT fine-tuné (VALIDATION) ===")
    print(f"macro-F1    : {summary['val_macro_f1']:.4f}   (baseline TF-IDF : 0.8006)")
    print(f"weighted-F1 : {summary['val_weighted_f1']:.4f}")
    print(f"accuracy    : {summary['val_accuracy']:.4f}")
    print(f"entraînement: {summary['train_seconds']:.0f}s sur {device}")
    print(f"\nModèle + metrics.json -> {out_dir}")
    log.info("transformer_done", **{k: summary[k] for k in ("val_macro_f1", "device")})


if __name__ == "__main__":
    main()
