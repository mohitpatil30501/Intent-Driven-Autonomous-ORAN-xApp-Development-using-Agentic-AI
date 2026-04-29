FROM python:3.12-slim

WORKDIR /app

# System dependencies for readability-lxml + CA certificates for SSL + git + envsubst
RUN apt-get update && apt-get install -y --no-install-recommends \
    libxml2-dev libxslt-dev gcc ca-certificates git gettext-base \
    && update-ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Clone the repository to a temporary folder and copy its contents
RUN git clone https://github.com/vkfolio/orio-search.git /tmp/repo \
    && cp -r /tmp/repo/app /app/app \
    && cp /tmp/repo/requirements.txt /app/ \
    && rm -rf /tmp/repo

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download FlashRank model to avoid first-request delay
RUN python -c "from flashrank import Ranker; Ranker(model_name='ms-marco-MiniLM-L-12-v2')" || true

# Config is mounted via docker-compose, so we don't need to copy it here
EXPOSE 8000

# Production: evaluate config.yaml with envsubst, then run gunicorn
CMD envsubst < /app/config.yaml > /tmp/config.yaml && \
    ORIO_SEARCH_CONFIG=/tmp/config.yaml \
    gunicorn app.main:app \
    --worker-class uvicorn.workers.UvicornWorker \
    --workers 4 \
    --bind 0.0.0.0:8000 \
    --timeout 60 \
    --graceful-timeout 30 \
    --keep-alive 5 \
    --access-logfile -
