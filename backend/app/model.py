import joblib

# Below this confidence, flag the ticket for human review instead of
# trusting the auto-classification. This is the difference between a
# toy classifier and a believable production pattern (human-in-the-loop).
CONFIDENCE_THRESHOLD = 0.6


class TicketClassifier:
    def __init__(self, model_path: str):
        self.pipeline = joblib.load(model_path)

    def predict(self, text: str) -> dict:
        proba = self.pipeline.predict_proba([text])[0]
        idx = proba.argmax()
        label = self.pipeline.classes_[idx]
        confidence = float(proba[idx])
        return {
            "label": label,
            "confidence": confidence,
            "low_confidence": confidence < CONFIDENCE_THRESHOLD,
        }
