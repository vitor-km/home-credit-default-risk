FROM python:3.13-slim

WORKDIR /app

COPY api/requirements.txt api/requirements.txt
RUN pip install --no-cache-dir -r api/requirements.txt

COPY api/ api/
COPY src/ src/
COPY models/scorecard_pipeline.json models/scorecard_pipeline.json

WORKDIR /app/api

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]