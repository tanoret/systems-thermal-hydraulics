# ── Stage 1: dependency layer ─────────────────────────────────────────────────
# Uses a slim Python image; pinned to a minor version for reproducible builds.
FROM python:3.11-slim AS deps

WORKDIR /app

# Install OS packages needed by numpy/iapws (BLAS headers, etc.) and then clean up.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt


# ── Stage 2: application image ────────────────────────────────────────────────
FROM python:3.11-slim AS app

WORKDIR /app

# Copy installed packages from the deps stage (keeps final image small).
COPY --from=deps /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=deps /usr/local/bin /usr/local/bin

# Copy the entire repo (src/, streamlit_app/, .streamlit/).
COPY . .

# Streamlit listens on this port; Cloud Run / Railway expect $PORT.
ENV PORT=8501
EXPOSE 8501

# Health-check so container orchestrators know when the app is ready.
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s \
    CMD curl -f http://localhost:${PORT}/_stcore/health || exit 1

# Use shell form so $PORT is expanded at runtime.
CMD streamlit run streamlit_app/app.py \
        --server.port=${PORT} \
        --server.address=0.0.0.0 \
        --server.headless=true \
        --server.runOnSave=false
