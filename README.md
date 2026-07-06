<div align="center">

# French Intent Classifier — fine-tuning CamemBERT industrialisé

**Du benchmark public contaminé au modèle quantifié servi en production :**
classification d'intentions en français (60 classes), avec des décisions prises **par la mesure** à chaque étape.

[![CI](https://github.com/behramkorkut/french-intent-classifier/actions/workflows/ci.yml/badge.svg)](https://github.com/behramkorkut/french-intent-classifier/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)
![uv](https://img.shields.io/badge/packaging-uv-DE5FE9?logo=astral&logoColor=white)
![CamemBERT](https://img.shields.io/badge/CamemBERT-fine--tuning-FFCC00)
![transformers](https://img.shields.io/badge/🤗_transformers-Trainer-FF9D00)
![ONNX](https://img.shields.io/badge/ONNX-INT8_per--channel-005CED?logo=onnx&logoColor=white)
![MLflow](https://img.shields.io/badge/MLflow-tracking-0194E2?logo=mlflow&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-serving-009688?logo=fastapi&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-sans_torch-2496ED?logo=docker&logoColor=white)
![pytest](https://img.shields.io/badge/tests-24-0A9EDC?logo=pytest&logoColor=white)

</div>

---

## Le problème

Le cœur d'un agent conversationnel sérieux : **router** la demande utilisateur
(« mets une alarme », « il va pleuvoir ? ») vers la bonne action. Un LLM généraliste sait le
faire — mais un modèle spécialisé fine-tuné est **plus rapide (4 ms), moins cher, auditable**,
et déployable souverainement. Données : [MASSIVE fr-FR](https://huggingface.co/datasets/SetFit/amazon_massive_intent_fr-FR)
(Amazon, ~11 500 phrases d'entraînement, 60 intentions réelles).

## Ce que ce projet démontre, chiffres à l'appui

### 1. Vérifier le benchmark au lieu de le croire

Des contrôles qualité automatiques (purs, testés) ont trouvé le split officiel **contaminé** :
**150 phrases du test et 98 de la validation présentes mot pour mot dans le train** (~5 % du
test), plus 373 doublons. Corrigé par déduplication + décontamination — en gardant les textes
identiques à labels différents (ambiguïté réelle du langage, pas un artefact). Sans ce
nettoyage, tous les scores ci-dessous seraient gonflés par de la mémorisation.

### 2. Une baseline honnête d'abord

TF-IDF + régression logistique (`class_weight=balanced`, 2,8 s de CPU) : la barre que le
transformer doit battre pour mériter son coût. Métrique primaire : **macro-F1** (déséquilibre
202x entre classes).

### 3. Le fine-tuning, justifié par la mesure (TEST descellé une seule fois)

| TEST (2 812 ex., décontaminé) | macro-F1 | weighted-F1 | accuracy |
|---|:---:|:---:|:---:|
| Baseline TF-IDF + logistique | 0,7584 | 0,8029 | 0,8019 |
| **CamemBERT fine-tuné** (8 epochs, T4) | **0,7990** | **0,8704** | **0,8730** |

**Δ +4,1 pts de macro-F1, +7,1 pts d'accuracy.** Lecture fine : la baseline s'effondre entre
validation et test (0,80 → 0,76) quand CamemBERT tient exactement (0,799) — généralisation
stable. Les confusions restantes sont sémantiquement cohérentes (`calendar_set` ↔
`calendar_query`) : le modèle échoue là où un humain hésiterait. Analyse d'erreurs complète
dans `reports/per_class_f1.csv`.

L'entraînement tourne sur Colab T4 mais **le notebook n'est qu'un exécuteur**
([notebooks/colab.md](notebooks/colab.md)) : toute la logique vit dans le paquet, versionnée,
testée, tracée dans MLflow (les runs sous-entraînés aussi — l'itération fait partie de l'histoire).

### 4. Optimiser, c'est mesurer

Cible de production : un VPS **CPU**. Export ONNX puis quantification INT8, avec **contrôle de
parité** systématique :

| moteur (CPU, batch=1) | macro-F1 val | p50 | p95 | taille |
|---|:---:|:---:|:---:|:---:|
| PyTorch fp32 | 0,7992 | 26,5 ms | 29,4 ms | 443 Mo |
| ONNX fp32 | 0,7992 | 11,3 ms | 15,0 ms | 443 Mo |
| **ONNX INT8 per-channel** | **0,7961** | **4,1 ms** | **5,7 ms** | **112 Mo** |

**x6,5 pour -0,3 point.** Le contrôle de parité a évité un piège : la quantification naïve
(per-tensor) perdait **9,6 points** de macro-F1 — rapide, léger, et silencieusement dégradé.
Personne ne l'aurait vu sans le harnais de mesure.

### 5. Servir léger

API FastAPI sur le moteur INT8 : `POST /predict` → intention + confiance + top-3 + latence.
L'image Docker de serving n'embarque **ni PyTorch ni le modèle fp32** — uniquement onnxruntime
et le tokenizer.

```bash
curl -s -X POST localhost:8000/predict -H "Content-Type: application/json" \
  -d '{"text": "mets une alarme à sept heures demain matin"}'
```
```json
{"intent": "alarm_set", "confidence": 0.9611,
 "top": [{"intent": "alarm_set", "confidence": 0.9611},
         {"intent": "alarm_query", "confidence": 0.0049},
         {"intent": "audio_volume_mute", "confidence": 0.0047}],
 "latency_ms": 4.9, "engine": "onnxruntime-int8-per-channel"}
```

## Quickstart

```bash
git clone https://github.com/behramkorkut/french-intent-classifier.git
cd french-intent-classifier
make setup          # uv sync
make data           # télécharge + contrôle + décontamine MASSIVE fr-FR
make baseline       # la barre à battre (~30 s CPU)
make train          # fine-tuning (GPU conseillé : notebooks/colab.md)
make evaluate       # verdict final baseline vs CamemBERT
make optimize       # ONNX + INT8 + benchmark de latence
make serve          # API -> http://localhost:8000/docs
make check          # tout ce que la CI vérifie
```

`make help` liste toutes les cibles.

## Structure

```
french-intent-classifier/
├── src/intent_classifier/
│   ├── common/config.py       # config typée (pydantic-settings), graine partagée
│   ├── data/load.py           # MASSIVE fr-FR + contrôles qualité + décontamination
│   ├── modeling/baseline.py   # TF-IDF + logistique (la barre à battre)
│   ├── modeling/transformer.py# fine-tuning CamemBERT (Trainer HF, MLflow)
│   ├── evaluation/compare.py  # verdict final : TEST descellé une seule fois
│   ├── optimization/          # export ONNX + INT8 + parité + benchmark p50/p95
│   └── serving/api.py         # FastAPI sur le moteur INT8
├── tests/                     # 24 tests (contrôles données, métriques, API…)
├── notebooks/colab.md         # Colab = simple exécuteur du paquet
├── Dockerfile                 # image de serving SANS torch
├── .github/workflows/ci.yml   # ruff + format + pytest
└── Makefile
```

## Discipline d'évaluation

Itération sur la **validation** uniquement ; le **test** est resté sous scellés jusqu'à la
comparaison finale, lue une seule fois. Métrique primaire macro-F1 (déséquilibre 202x). Chaque
run — y compris les ratés — est tracé dans MLflow (`make mlflow`).

## Licence

MIT.
