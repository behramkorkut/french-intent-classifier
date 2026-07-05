"""Tests des contrôles qualité — fonction pure, aucun téléchargement."""

import pandas as pd

from intent_classifier.data.load import sanity_checks


def _frames(train_texts, val_texts, test_texts, intent="greet"):
    def df(texts):
        return pd.DataFrame({"text": texts, "label": 0, "intent": intent})

    return {"train": df(train_texts), "validation": df(val_texts), "test": df(test_texts)}


def test_clean_frames_pass():
    frames = _frames(["bonjour", "salut"], ["coucou"], ["hello"])
    assert sanity_checks(frames) == []


def test_cross_split_contamination_is_detected():
    frames = _frames(["bonjour", "salut"], ["bonjour"], ["hello"])
    assert any("contamination" in i for i in sanity_checks(frames))


def test_internal_duplicates_are_detected():
    frames = _frames(["bonjour", "bonjour"], ["coucou"], ["hello"])
    assert any("doublons internes" in i for i in sanity_checks(frames))


def test_empty_text_is_detected():
    frames = _frames(["bonjour", "  "], ["coucou"], ["hello"])
    assert any("textes vides" in i for i in sanity_checks(frames))


def test_unseen_intent_in_test_is_detected():
    frames = _frames(["bonjour"], ["coucou"], ["hello"])
    frames["test"]["intent"] = "unknown_intent"
    assert any("absentes du train" in i for i in sanity_checks(frames))


def test_clean_frames_removes_duplicates_and_contamination():
    from intent_classifier.data.load import clean_frames

    frames = _frames(
        ["bonjour", "bonjour", "salut"],  # 1 doublon interne
        ["bonjour", "coucou"],  # 1 contamination
        ["salut", "hello", "hello"],  # 1 contamination + 1 doublon
    )
    cleaned, stats = clean_frames(frames)
    assert stats["train_doublons"] == 1
    assert stats["validation_contamination"] == 1
    assert stats["test_contamination"] == 1
    assert list(cleaned["validation"]["text"]) == ["coucou"]
    assert list(cleaned["test"]["text"]) == ["hello"]
    # après nettoyage, plus aucune contamination détectable
    from intent_classifier.data.load import sanity_checks

    assert not any("contamination" in i for i in sanity_checks(cleaned))


def test_clean_frames_keeps_genuine_ambiguity():
    import pandas as pd

    from intent_classifier.data.load import clean_frames

    # même texte, deux labels différents = ambiguïté réelle, on garde
    train = pd.DataFrame(
        {"text": ["stop", "stop"], "label": [0, 1], "intent": ["mute", "cancel"]}
    )
    frames = {
        "train": train,
        "validation": pd.DataFrame({"text": ["ok"], "label": [0], "intent": ["mute"]}),
        "test": pd.DataFrame({"text": ["ko"], "label": [0], "intent": ["mute"]}),
    }
    cleaned, stats = clean_frames(frames)
    assert len(cleaned["train"]) == 2
    assert stats["train_doublons"] == 0
