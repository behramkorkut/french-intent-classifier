# French Intent Classifier — fine-tuning CamemBERT industrialisé

Classification d'**intentions utilisateur en français** (60 classes, dataset réel
[MASSIVE fr-FR](https://huggingface.co/datasets/AmazonScience/massive) d'Amazon) par
fine-tuning de **CamemBERT** — avec la même exigence d'industrialisation que mes autres
projets : baseline honnête, suivi d'expériences, tests, optimisation d'inférence, API.

> 🚧 Projet en cours — feuille de route ci-dessous.

## Pourquoi ce projet

Le cœur d'un agent conversationnel sérieux : router la demande de l'utilisateur
(« je veux résilier », « c'est quoi la garantie X ? ») vers la bonne action. Un LLM
généraliste sait le faire, mais un modèle fine-tuné spécialisé est **plus rapide, moins
cher, auditable** — et déployable souverainement.

## Feuille de route

1. ✅ Scaffold (uv, src layout, config typée, tests, lint)
2. Données : MASSIVE fr-FR — EDA, distribution des 60 intentions
3. **Baseline TF-IDF + régression logistique** — le transformer devra la battre pour mériter sa place
4. Fine-tuning CamemBERT (GPU T4 Colab — le notebook n'est qu'un *exécuteur* du paquet)
5. Évaluation : macro-F1, matrice de confusion, analyse d'erreurs
6. Optimisation d'inférence : export **ONNX** + quantification INT8, benchmark de latence
7. API FastAPI + Docker + CI
8. Déploiement production (VPS OVHcloud)

## Démarrage

```bash
uv sync
uv run pytest -q
uv run ruff check .
```
