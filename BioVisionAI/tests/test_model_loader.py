import numpy as np

from utils import model_loader


def test_missing_model_returns_non_diagnostic_marker(monkeypatch):
    monkeypatch.setattr(
        model_loader,
        "get_model",
        lambda category: (None, {0: "Healthy", 1: "Disease"}),
    )

    result = model_loader.predict("plant", np.zeros((1, 224, 224, 3)))

    assert result == [("DEMO_MODE_NO_TRAINED_MODEL", 0.0)]


def test_model_failure_does_not_return_a_confident_disease(monkeypatch):
    class BrokenModel:
        def predict(self, image, verbose=0):
            raise RuntimeError("invalid model output")

    monkeypatch.setattr(
        model_loader,
        "get_model",
        lambda category: (BrokenModel(), {0: "Disease"}),
    )

    result = model_loader.predict("plant", np.zeros((1, 224, 224, 3)))

    assert result == [("PREDICTION_UNAVAILABLE_CHECK_MODEL", 0.0)]



def test_model_exists_requires_a_loadable_trained_model(monkeypatch):
    monkeypatch.setattr(
        model_loader,
        "get_model",
        lambda category: (object(), {0: "Healthy"}),
    )
    assert model_loader.model_exists("plant") is True

    monkeypatch.setattr(
        model_loader,
        "get_model",
        lambda category: (None, {0: "Healthy"}),
    )
    assert model_loader.model_exists("plant") is False
