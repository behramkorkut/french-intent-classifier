"""Tests des analyses d'évaluation — fonctions pures, aucun modèle chargé."""

from intent_classifier.evaluation.compare import metric_table, top_confusions, worst_classes

Y_TRUE = ["a", "a", "b", "b", "c", "c"]
Y_PRED = ["a", "a", "b", "c", "b", "c"]  # b et c confondus une fois chacun


def test_metric_table_shape_and_bounds():
    table = metric_table(Y_TRUE, {"m1": Y_PRED, "m2": Y_TRUE})
    assert list(table.index) == ["m1", "m2"]
    assert table.loc["m2", "macro_f1"] == 1.0
    assert 0.0 < table.loc["m1", "macro_f1"] < 1.0


def test_worst_classes_sorted_ascending_with_support():
    worst = worst_classes(Y_TRUE, Y_PRED, k=3)
    assert list(worst.columns) == ["intent", "f1", "support"]
    assert worst["f1"].is_monotonic_increasing
    assert worst.iloc[-1]["intent"] == "a"  # la classe parfaite arrive en dernier


def test_top_confusions_excludes_diagonal():
    conf = top_confusions(Y_TRUE, Y_PRED, k=5)
    assert all(conf["vraie"] != conf["prédite"])
    pairs = set(zip(conf["vraie"], conf["prédite"], strict=True))
    assert ("b", "c") in pairs and ("c", "b") in pairs
