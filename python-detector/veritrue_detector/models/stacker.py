from __future__ import annotations

from dataclasses import dataclass


def load_stacker(path: str | None):
    if not path:
        return None
    try:
        import joblib

        model = joblib.load(path)
        return SklearnStacker(model)
    except Exception:
        return None


@dataclass
class SklearnStacker:
    model: object

    def predict_proba_ai(self, probs: dict[str, float]) -> float:
        # Deterministic feature order
        keys = ["efficientnet", "vit", "clip"]
        x = [[float(probs.get(k, 0.5)) for k in keys]]

        if hasattr(self.model, "predict_proba"):
            p = self.model.predict_proba(x)
            # assume class 1 = AI
            return float(p[0][1])

        if hasattr(self.model, "decision_function"):
            import math

            z = float(self.model.decision_function(x)[0])
            return 1.0 / (1.0 + math.exp(-z))

        return float(sum(x[0]) / len(x[0]))
