"""Optimisation d'inférence : export ONNX + quantification INT8 + benchmark.

Pourquoi : la cible de production est un VPS **CPU** (pas de GPU). Deux leviers :

1. **Export ONNX** — graphe figé, exécuté par onnxruntime (C++), sans l'overhead
   Python/PyTorch.
2. **Quantification dynamique INT8** — les poids passent de float32 à int8
   (~4x plus petit), les calculs profitent des instructions vectorielles CPU.

Deux vérifications, car optimiser sans mesurer serait de la superstition :
- **Parité de qualité** : macro-F1 sur la VALIDATION pour chaque variante
  (le TEST reste sous scellés — la parité est un choix d'ingénierie, pas une
  évaluation finale).
- **Benchmark de latence** batch=1 — le cas réel d'un chatbot — p50/p95 après
  échauffement, plus la taille des artefacts sur disque.

Note archi : les kernels INT8 diffèrent selon le CPU (arm64 = ce Mac,
avx2 = le VPS x86). L'arch est auto-détectée, surchargée par --arch pour
préparer un artefact destiné à une autre machine.
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import time
from collections.abc import Callable

import numpy as np
import structlog
from sklearn.metrics import f1_score

from intent_classifier.common.config import settings
from intent_classifier.modeling.baseline import load_split

log = structlog.get_logger()


# ---------- Mesure (pur, testable) ----------
def percentiles(samples_ms: list[float]) -> dict[str, float]:
    return {
        "p50_ms": round(statistics.median(samples_ms), 2),
        "p95_ms": round(float(np.percentile(samples_ms, 95)), 2),
        "mean_ms": round(statistics.fmean(samples_ms), 2),
    }


def bench(fn: Callable[[str], object], payloads: list[str], warmup: int = 20) -> dict[str, float]:
    """Latence par appel (ms), batch=1, après échauffement."""
    for text in payloads[:warmup]:
        fn(text)
    times: list[float] = []
    for text in payloads:
        t0 = time.perf_counter()
        fn(text)
        times.append((time.perf_counter() - t0) * 1000)
    return percentiles(times)


def size_mb(path) -> float:
    return round(path.stat().st_size / 1e6, 1)


def main() -> None:
    import mlflow
    import torch
    from optimum.onnxruntime import ORTModelForSequenceClassification, ORTQuantizer
    from optimum.onnxruntime.configuration import AutoQuantizationConfig
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    parser = argparse.ArgumentParser(description="Export ONNX + quantification + benchmark")
    parser.add_argument(
        "--arch",
        choices=["arm64", "avx2"],
        default=None,
        help="Kernels INT8 cibles (défaut : auto — arm64 sur ce Mac, avx2 pour le VPS x86)",
    )
    parser.add_argument("--n-bench", type=int, default=200, help="Appels mesurés par variante")
    args = parser.parse_args()

    model_dir = settings.models_dir / "camembert"
    onnx_dir = settings.models_dir / "onnx"
    tok = AutoTokenizer.from_pretrained(model_dir)

    # --- 1) Export ONNX (fp32) ---
    log.info("onnx_export_start")
    ort_fp32 = ORTModelForSequenceClassification.from_pretrained(model_dir, export=True)
    ort_fp32.save_pretrained(onnx_dir)
    tok.save_pretrained(onnx_dir)

    # --- 2) Quantification dynamique INT8 ---
    arch = args.arch or ("arm64" if platform.machine() == "arm64" else "avx2")
    # per_channel=True : une échelle de quantification par canal de poids plutôt
    # que par tenseur — indispensable ici : en per-tensor, l'INT8 perdait
    # 9,6 points de macro-F1 (0,799 -> 0,703), mesuré par le contrôle de parité.
    qconfig = (
        AutoQuantizationConfig.arm64(is_static=False, per_channel=True)
        if arch == "arm64"
        else AutoQuantizationConfig.avx2(is_static=False, per_channel=True)
    )
    # file_name explicite : après un premier run, le dossier contient aussi
    # model_quantized.onnx — sans ça, ORTQuantizer refuse de choisir.
    ORTQuantizer.from_pretrained(onnx_dir, file_name="model.onnx").quantize(
        save_dir=onnx_dir, quantization_config=qconfig
    )
    ort_int8 = ORTModelForSequenceClassification.from_pretrained(
        onnx_dir, file_name="model_quantized.onnx"
    )
    log.info("quantization_done", arch=arch)

    # --- 3) Moteurs à comparer (CPU partout : la cible de prod) ---
    torch_cpu = AutoModelForSequenceClassification.from_pretrained(model_dir).eval()
    id2label = torch_cpu.config.id2label

    def make_predict(model) -> Callable[[str], str]:
        def predict(text: str) -> str:
            enc = tok(text, return_tensors="pt", truncation=True, max_length=64)
            with torch.no_grad():
                logits = model(**enc).logits
            return id2label[int(logits.argmax(-1))]

        return predict

    engines: dict[str, Callable[[str], str]] = {
        "torch_fp32_cpu": make_predict(torch_cpu),
        "onnx_fp32": make_predict(ort_fp32),
        "onnx_int8": make_predict(ort_int8),
    }

    # --- 4) Parité de qualité (VALIDATION) + latence ---
    val = load_split("validation")
    y_val = val["intent"].tolist()
    payloads = val["text"].tolist()[: args.n_bench]

    results: dict[str, dict] = {}
    for name, predict in engines.items():
        preds = [predict(t) for t in val["text"]]
        macro = float(f1_score(y_val, preds, average="macro"))
        latency = bench(predict, payloads)
        results[name] = {"val_macro_f1": round(macro, 4), **latency}
        log.info("engine_done", engine=name, **results[name])

    results["torch_fp32_cpu"]["size_mb"] = size_mb(model_dir / "model.safetensors")
    results["onnx_fp32"]["size_mb"] = size_mb(onnx_dir / "model.onnx")
    results["onnx_int8"]["size_mb"] = size_mb(onnx_dir / "model_quantized.onnx")

    # --- 5) Rapport ---
    speedup = results["torch_fp32_cpu"]["p50_ms"] / results["onnx_int8"]["p50_ms"]
    print(
        f"\n=== Benchmark d'inférence (CPU {platform.machine()}, batch=1, "
        f"n={args.n_bench}, kernels {arch}) ==="
    )
    header = f"{'moteur':<16} {'macro-F1 val':>12} {'p50 ms':>8} {'p95 ms':>8} {'taille Mo':>10}"
    print(header)
    for name, r in results.items():
        print(
            f"{name:<16} {r['val_macro_f1']:>12.4f} {r['p50_ms']:>8.2f} "
            f"{r['p95_ms']:>8.2f} {r['size_mb']:>10.1f}"
        )
    print(f"\nSpeed-up p50 (torch fp32 -> onnx int8) : x{speedup:.1f}")

    settings.reports_dir.mkdir(parents=True, exist_ok=True)
    (settings.reports_dir / "latency_benchmark.json").write_text(
        json.dumps({"arch": arch, "machine": platform.machine(), **results}, indent=2)
    )

    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment(settings.experiment_name)
    with mlflow.start_run(run_name=f"onnx-optimization-{arch}"):
        mlflow.log_param("arch", arch)
        for name, r in results.items():
            mlflow.log_metrics({f"{name}_{k}": v for k, v in r.items()})

    print(f"Rapport -> {settings.reports_dir / 'latency_benchmark.json'}")


if __name__ == "__main__":
    main()
