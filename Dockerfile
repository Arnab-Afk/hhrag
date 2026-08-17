FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential git \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Build the demo index at image build so Spaces boots with a live corpus.
# Swap this for a prebuilt MSMARCO-XI shard when you have subsampled.
RUN python -m ingest.build_index --demo

ENV PORT=7860
EXPOSE 7860
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]
