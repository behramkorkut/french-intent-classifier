"""Tests des briques pures du fine-tuning — sans GPU ni téléchargement de modèle."""

import numpy as np
import pandas as pd

from intent_classifier.modeling.transformer import build_label_maps, compute_metrics


def test_label_maps_are_deterministic_and_inverse():
    train = pd.DataFrame({"intent": ["play_music", "alarm_set", "play_music"]})
    label2id, id2label = build_label_maps(train)
    assert label2id == {"alarm_set": 0, "play_music": 1}  # tri alphabétique
    assert {i: n for n, i in label2id.items()} == id2label


def test_compute_metrics_perfect_prediction():
    logits = np.array([[5.0, 0.0], [0.0, 5.0], [4.0, 1.0]])
    labels = np.array([0, 1, 0])
    m = compute_metrics((logits, labels))
    assert m["macro_f1"] == 1.0
    assert m["accuracy"] == 1.0


def test_compute_metrics_bounded_on_errors():
    logits = np.array([[5.0, 0.0], [5.0, 0.0]])  # prédit toujours 0
    labels = np.array([0, 1])
    m = compute_metrics((logits, labels))
    assert 0.0 < m["accuracy"] < 1.0
    assert 0.0 < m["macro_f1"] < 1.0
