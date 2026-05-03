FROM python:3.10-slim

WORKDIR /app/server

RUN apt-get update && apt-get install -y build-essential && rm -rf /var/lib/apt/lists/*

COPY server/requirements.txt .
COPY server/requirements-server.txt .

RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir -r requirements-server.txt

COPY server/ .

ENV FLEXRIC_INDEX_DIR=/app/flexric_index/index

EXPOSE 7070

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "7070"]
