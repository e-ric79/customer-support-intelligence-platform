from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from prometheus_fastapi_instrumentator import Instrumentator

from .model import TicketClassifier
from .schemas import TicketRequest, TicketResponse

MODEL_PATH = "model_artifacts/model.pkl"
classifier: TicketClassifier | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Loaded once at process startup, not per-request — a fresh
    # load per request would be slow and wasteful.
    global classifier
    classifier = TicketClassifier(MODEL_PATH)
    yield


app = FastAPI(title="Ticket Classifier API", version="0.1.0", lifespan=lifespan)

# Exposes /metrics for Prometheus to scrape — same pattern you used
# in swahili-sentiment-mlops.
Instrumentator().instrument(app).expose(app)


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": classifier is not None}


@app.post("/predict", response_model=TicketResponse)
def predict(req: TicketRequest):
    if classifier is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    result = classifier.predict(req.text)
    return TicketResponse(**result)
