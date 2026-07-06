"""Chargement de MASSIVE fr-FR + contrôles de qualité du split officiel.

Pourquoi contrôler un split « officiel » : un benchmark public n'est pas une
garantie. Les contaminations classiques — phrases identiques présentes à la fois
en train et en test, doublons internes, classes absentes d'un split — gonflent
artificiellement les scores. On vérifie au lieu de croire (le même réflexe que
le split par patient sur readmission-risk-ml).

Sorties : data/{train,validation,test}.parquet + reports/intent_distribution.png.
"""

from __future__ import annotations

import os

# Backend non interactif, forcé AVANT l'import : Colab exporte
# MPLBACKEND=module://matplotlib_inline.backend_inline, un backend propre à son
# kernel que notre venv ne connaît pas — l'import de matplotlib planterait.
os.environ["MPLBACKEND"] = "Agg"

import matplotlib.pyplot as plt
import pandas as pd
import structlog
from datasets import load_dataset

from intent_classifier.common.config import settings

log = structlog.get_logger()

SPLITS = ("train", "validation", "test")


def to_frames() -> dict[str, pd.DataFrame]:
    """Télécharge MASSIVE fr-FR et renvoie un DataFrame par split.

    Colonnes normalisées : `text` (la phrase), `label` (id entier),
    `intent` (nom lisible, ex. `alarm_set`).
    """
    ds = load_dataset(settings.dataset_name)
    frames: dict[str, pd.DataFrame] = {}
    for split in SPLITS:
        df = ds[split].to_pandas()[[settings.text_col, settings.label_col, settings.intent_col]]
        df = df.rename(
            columns={
                settings.text_col: "text",
                settings.label_col: "label",
                settings.intent_col: "intent",
            }
        )
        frames[split] = df
    return frames


def sanity_checks(frames: dict[str, pd.DataFrame]) -> list[str]:
    """Contrôles anti-contamination. Renvoie la liste des problèmes détectés.

    Fonction pure (testable sans réseau) : elle ne télécharge rien.
    """
    issues: list[str] = []

    for split, df in frames.items():
        if df["text"].str.strip().eq("").any():
            issues.append(f"{split}: textes vides")
        n_dup = int(df["text"].duplicated().sum())
        if n_dup:
            issues.append(f"{split}: {n_dup} doublons internes")

    # Contamination inter-splits : la même phrase en train ET en éval.
    train_texts = set(frames["train"]["text"])
    for split in ("validation", "test"):
        n_leak = int(frames[split]["text"].isin(train_texts).sum())
        if n_leak:
            issues.append(f"{split}: {n_leak} phrases identiques au train (contamination)")

    # Couverture : une intention jamais vue au train est inapprenable.
    train_intents = set(frames["train"]["intent"])
    for split in ("validation", "test"):
        missing = sorted(set(frames[split]["intent"]) - train_intents)
        if missing:
            issues.append(f"{split}: intentions absentes du train: {missing[:5]}")

    return issues


def clean_frames(
    frames: dict[str, pd.DataFrame],
) -> tuple[dict[str, pd.DataFrame], dict[str, int]]:
    """Déduplication + décontamination. Renvoie (frames propres, stats de retrait).

    - Doublons internes exacts (même texte, même label) : retirés partout.
    - Contamination : toute phrase d'évaluation présente dans le train est
      retirée de validation/test — sinon on mesure de la mémorisation, pas de
      la généralisation.
    - Un même texte avec DEUX labels différents n'est PAS retiré : c'est de
      l'ambiguïté réelle du langage, pas un artefact.
    """
    out: dict[str, pd.DataFrame] = {}
    stats: dict[str, int] = {}
    for split, df in frames.items():
        before = len(df)
        out[split] = df.drop_duplicates(subset=["text", "label"], keep="first").reset_index(
            drop=True
        )
        stats[f"{split}_doublons"] = before - len(out[split])

    train_texts = set(out["train"]["text"])
    for split in ("validation", "test"):
        before = len(out[split])
        out[split] = out[split][~out[split]["text"].isin(train_texts)].reset_index(drop=True)
        stats[f"{split}_contamination"] = before - len(out[split])
    return out, stats


def _plot_distribution(train: pd.DataFrame, path) -> None:
    counts = train["intent"].value_counts().sort_values()
    fig, ax = plt.subplots(figsize=(8, 14))
    ax.barh(counts.index, counts.to_numpy(), color="#1f77b4")
    ax.set_xlabel("Exemples (train)")
    ax.set_title(f"MASSIVE fr-FR — {len(counts)} intentions")
    ax.tick_params(axis="y", labelsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    frames = to_frames()

    print("\n=== MASSIVE fr-FR ===")
    for split, df in frames.items():
        print(f"{split:<12} {len(df):>6} exemples · {df['intent'].nunique()} intentions")

    counts = frames["train"]["intent"].value_counts()
    print(f"\nClasse la plus fréquente : {counts.index[0]} ({counts.iloc[0]})")
    print(f"Classe la plus rare      : {counts.index[-1]} ({counts.iloc[-1]})")
    print(f"Ratio de déséquilibre    : {counts.iloc[0] / counts.iloc[-1]:.1f}x")

    issues = sanity_checks(frames)
    if issues:
        print("\n⚠ Contrôles qualité (split officiel, avant nettoyage) :")
        for issue in issues:
            print(f"  - {issue}")
    else:
        print("\n✓ Contrôles qualité : aucun problème détecté.")

    frames, stats = clean_frames(frames)
    print("\n=== Nettoyage ===")
    for key, n in stats.items():
        if n:
            print(f"  - {key}: {n} lignes retirées")
    print("Tailles finales : " + " · ".join(f"{s}={len(df)}" for s, df in frames.items()))

    settings.data_dir.mkdir(parents=True, exist_ok=True)
    settings.reports_dir.mkdir(parents=True, exist_ok=True)
    for split, df in frames.items():
        df.to_parquet(settings.data_dir / f"{split}.parquet", index=False)
    _plot_distribution(frames["train"], settings.reports_dir / "intent_distribution.png")

    print(f"\nParquets -> {settings.data_dir}")
    print(f"Distribution -> {settings.reports_dir / 'intent_distribution.png'}")
    log.info("data_ready", **{s: len(df) for s, df in frames.items()})


if __name__ == "__main__":
    main()
