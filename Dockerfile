FROM python:3.11-slim

WORKDIR /app

# Layer-cache dependencies separately from source code
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

# 4 workers; remove --workers flag if deploying to a single-vCPU free tier
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
