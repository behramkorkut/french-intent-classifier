"""La baseline s'entraîne, prédit et s'évalue — sur un micro-corpus synthétique."""

import pandas as pd

from intent_classifier.modeling.baseline import build_pipeline, evaluate

_TRAIN = pd.DataFrame(
    {
        "text": ["règle une alarme demain matin"] * 5 + ["joue de la musique jazz"] * 5,
        "intent": ["alarm_set"] * 5 + ["play_music"] * 5,
    }
)
_VAL = pd.DataFrame(
    {
        "text": ["mets une alarme demain", "joue du jazz"],
        "intent": ["alarm_set", "play_music"],
    }
)


def test_pipeline_learns_and_predicts():
    model = build_pipeline(min_df=1)
    model.fit(_TRAIN["text"], _TRAIN["intent"])
    assert model.predict(["règle une alarme"])[0] == "alarm_set"


def test_evaluate_returns_bounded_metrics():
    model = build_pipeline(min_df=1)
    model.fit(_TRAIN["text"], _TRAIN["intent"])
    metrics = evaluate(model, _VAL)
    assert set(metrics) == {"macro_f1", "weighted_f1", "accuracy"}
    assert all(0.0 <= v <= 1.0 for v in metrics.values())
    assert metrics["macro_f1"] == 1.0  # corpus trivialement séparable
