FROM python:3.11-slim

WORKDIR /srv/app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ app/
COPY scripts/ scripts/
COPY evaluation/ evaluation/
COPY data/raw/ data/raw/
COPY .env.example .env.example

EXPOSE 8000

# Builds the index on first start (cached afterwards via the /srv/app/data
# volume - see docker-compose.yml) then serves the API.
CMD ["sh", "-c", "python scripts/build_index.py && uvicorn app.main:app --host 0.0.0.0 --port 8000"]
