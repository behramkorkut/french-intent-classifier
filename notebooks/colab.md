# Entraînement sur Colab (GPU T4) — le notebook n'est qu'un exécuteur

Ouvre [colab.research.google.com](https://colab.research.google.com) → Nouveau notebook →
**Exécution → Modifier le type d'exécution → T4 GPU**. Puis une cellule par bloc :

```python
# 1. GPU présent ?
!nvidia-smi | head -12
```

```python
# 2. Cloner le paquet — toute la logique est dans le repo, rien dans le notebook
!git clone https://github.com/behramkorkut/french-intent-classifier.git
%cd french-intent-classifier
```

```python
# 3. Environnement (uv résout torch CUDA pour la T4)
!pip install -q uv
!uv sync
```

```python
# 4. Données (téléchargement + contrôles qualité + décontamination)
!uv run intents-data
```

```python
# 5. Fine-tuning (~10-15 min sur T4)
!uv run intents-train-transformer --epochs 4
```

```python
# 6. Rapatrier le modèle
!zip -rq camembert.zip models/camembert
from google.colab import files
files.download("camembert.zip")
```

De retour sur le Mac : dézipper dans `models/camembert/` à la racine du projet.

> Smoke test local (Mac, ~2 min, sous-échantillon) avant de consommer du Colab :
> `uv run intents-train-transformer --epochs 1 --max-samples 500`
