"""Tests des briques de mesure — pures, sans modèle ni export."""

import time

from intent_classifier.optimization.onnx_export import bench, percentiles


def test_percentiles_keys_and_order():
    p = percentiles([1.0, 2.0, 3.0, 4.0, 100.0])
    assert set(p) == {"p50_ms", "p95_ms", "mean_ms"}
    assert p["p50_ms"] <= p["p95_ms"]
    assert p["p50_ms"] == 3.0


def test_bench_measures_a_sleeping_function():
    def slow(_: str) -> None:
        time.sleep(0.001)  # ~1 ms

    result = bench(slow, ["x"] * 30, warmup=5)
    assert result["p50_ms"] >= 1.0  # au moins le sleep
    assert result["p95_ms"] < 50.0  # et pas un ordre de grandeur au-dessus
