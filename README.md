# Customer support ticket classifier

An ML-powered support ticket classifier: a trained model served through a FastAPI
backend, consumed by a Streamlit UI, both containerized with Docker, orchestrated
on Kubernetes with health checks and replicas, and shipped through a GitHub
Actions pipeline that runs tests before building and pushing images.

## Architecture

```
GitHub push → GitHub Actions (test → build → push) → image registry
                                                            ↓
                                          Kubernetes cluster (Minikube)
                                          ┌─────────────────────────┐
                                          │ Streamlit  →  FastAPI    │
                                          │ (frontend)   (backend)   │
                                          │              ↓           │
                                          │         model.pkl        │
                                          └─────────────────────────┘
```

## Dataset and model

- **Source**: [`bitext/Bitext-customer-support-llm-chatbot-training-dataset`](https://huggingface.co/datasets/bitext/Bitext-customer-support-llm-chatbot-training-dataset)
  on Hugging Face — an intent-classification dataset where `category` is
  genuinely derived from the customer utterance.
- **Switched from an earlier 8k-row Kaggle-style dataset** after diagnostics
  showed its `Ticket Type` field was not actually derived from the ticket
  text — the baseline model scored macro-F1 ≈ 0.19 against a random-chance
  floor of 0.20 on 5 balanced classes, with a uniform confusion matrix (the
  signature of an unlearnable label). Caught this before wasting time on
  transformer upgrades that couldn't have helped, since the problem was the
  data contract, not the model.
- **Target**: `category` — exact class list and counts printed by
  `prepare_data.py` on first run (check for imbalance before training;
  stratified split handles moderate imbalance, but heavy imbalance would
  call for `class_weight="balanced"`, already set in `train.py`).
- **Input**: the customer `instruction` text only.
- **PII**: this dataset is synthetic instruction/response pairs, not real
  customer records — verify with a fresh check whenever swapping in a new
  dataset, don't assume.
- Baseline model: TF-IDF (unigrams + bigrams) + Logistic Regression.
  <!-- TODO: replace with real macro-F1 once trained -->

## Running locally

```bash
# 1. Train the model (produces backend/model_artifacts/model.pkl)
pip install -r training/requirements.txt
python training/prepare_data.py
python training/train.py
python training/evaluate.py

# 2. Run the backend
cd backend
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000

# 3. Run the frontend (separate terminal)
cd frontend
pip install -r requirements.txt
API_URL=http://localhost:8000 streamlit run streamlit_app.py
```

## Running on Kubernetes (Minikube)

```bash
minikube start
eval $(minikube docker-env)   # build images directly into Minikube's registry

docker build -t ticket-classifier-backend:latest ./backend
docker build -t ticket-classifier-frontend:latest ./frontend

kubectl apply -f k8s/backend.yaml
kubectl apply -f k8s/frontend.yaml

minikube service frontend-service
```

## Testing

```bash
cd backend && python -m pytest tests/ -v
```

## CI/CD

`.github/workflows/ci-cd.yaml` runs backend tests on every push and PR to
`main`. Only if tests pass does it build and push both Docker images to
GHCR — the pipeline never ships an untested image.

## Design decisions worth knowing for a walkthrough

- **Backend and frontend are separate services communicating over HTTP**,
  not a monolith — this is what makes independent scaling, deployment,
  and the Kubernetes Service/DNS story meaningful.
- **Confidence threshold + `low_confidence` flag**: predictions below 60%
  confidence are flagged for human review rather than trusted blindly —
  a basic human-in-the-loop safety net.
- **Class weights are balanced in training** even though the dataset is
  already close to balanced — cheap insurance, no downside.
- **Cloud is intentionally deferred.** Minikube proves every Kubernetes
  concept (Deployments, Services, health probes, replicas) without cloud
  cost or complexity. Next step: point CI/CD at EKS/AKS/GKE.

## Known limitations

- The source dataset is synthetic/templated, so real-world accuracy is
  likely lower than test-set accuracy — validate against a small hand-written
  set of realistic tickets before trusting the numbers.
- Baseline model is classical (TF-IDF + Logistic Regression); a fine-tuned
  transformer is a natural v2 once the full pipeline is proven end-to-end.
