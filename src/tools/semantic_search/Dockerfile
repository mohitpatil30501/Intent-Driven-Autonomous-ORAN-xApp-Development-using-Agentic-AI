FROM python:3.10-slim

# Install Git and RipGrep (for exact keyword searching)
RUN apt-get update && apt-get install -y git ripgrep && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]