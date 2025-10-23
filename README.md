# Livesensor-ML

This repository provides a FastAPI service around a sensor-failure model and a small data pipeline.

Quick start
-----------
1. Create a virtual environment and install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. Provide MongoDB URI via environment variable (recommended):

```bash
export MONGODB_URI="mongodb+srv://<user>:<pass>@cluster.example/mydb"
```

(Alternatively, if you use the Streamlit UI, place a `.streamlit/secrets.toml` with `MONGODB_URI = "..."`.)

3. Run the API:

```bash
python3 main.py
```

The API will bind to the host/port defined in `sensor/constant/application.py` (default `0.0.0.0:8080`).

Endpoints
---------
- GET / -> redirect to /docs (FastAPI Swagger UI)
- GET /train -> start a training pipeline run (blocking call)
- POST /predict -> upload a CSV file (form field `file`) and receive predictions as CSV
- GET /health -> simple health check (checks model file and MongoDB connectivity)

Notes
-----
- For local dev, prefer using `MONGODB_URI` env var. Streamlit secrets are supported as a fallback but are optional.
- If you want to use the Streamlit app (older behavior), run `streamlit run main.py` but avoid port conflicts (8080). The project previously used Streamlit and the devcontainer runs streamlit by default; change the devcontainer settings if necessary.

Streamlit
---------

This repository now includes a lightweight Streamlit demo UI at `streamlit_app.py` for interactive prediction and quick testing.

Two deployment options:

1) FastAPI backend (default): keep `main.py` as the API server and use `streamlit_app.py` locally as a client UI (it loads the model from disk directly).

2) Streamlit-only: run `streamlit run streamlit_app.py` to launch a single-process UI that loads the trained model from `SAVED_MODEL_DIR`.

Run locally (Streamlit):

```bash
# Create and activate a virtualenv, install dependencies from requirements.txt
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Start the Streamlit demo
streamlit run streamlit_app.py --server.enableCORS false --server.enableXsrfProtection false
```

Streamlit Cloud / Heroku deployment
-----------------------------------

For Streamlit Cloud or Heroku-like platforms, add a `Procfile` at the repository root with the following content:

```
web: streamlit run streamlit_app.py --server.port $PORT --server.enableCORS false
```

Provide your `MONGODB_URI` via Streamlit secrets (`.streamlit/secrets.toml`) or as an environment variable in the deployment platform. An example template is provided in `.streamlit/secrets.toml.example`.

Secrets and MongoDB
--------------------

If you want the app to use MongoDB, set `MONGODB_URI` in your environment or add it to Streamlit secrets (see `.streamlit/secrets.toml.example`). Without a MongoDB URI the app will fall back to an in-memory store for local development.

Docker quickstart (easy & fast)
--------------------------------

If you have Docker installed, you can build and run the full stack (API + MongoDB) quickly with Docker Compose.

Build and start the stack:

```bash
docker compose up --build -d
```

Check logs for the API service:

```bash
docker compose logs -f api
```

Stop the stack:

```bash
docker compose down
```

Then verify the health endpoint:

```bash
curl http://localhost:8080/health
```


If you'd like, I can add a Dockerfile, unit tests, or CI setup next.
