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

If you'd like, I can add a Dockerfile, unit tests, or CI setup next.
