FROM python:3.11-slim

WORKDIR /app

# Layer-cache dependencies separately from source code
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 7860

# Runs uvicorn on port 7860 (Hugging Face default)
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "7860"]
