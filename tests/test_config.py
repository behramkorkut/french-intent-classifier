"""Le socle est en place : la config se charge et porte les bonnes valeurs."""

from intent_classifier.common.config import PROJECT_ROOT, settings


def test_settings_defaults():
    assert settings.random_seed == 42
    assert settings.dataset_config == "fr-FR"
    assert settings.transformer_model == "camembert-base"


def test_paths_are_anchored_at_project_root():
    assert PROJECT_ROOT.name == "french-intent-classifier"
    assert settings.data_dir == PROJECT_ROOT / "data"
