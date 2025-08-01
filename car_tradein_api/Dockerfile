# ---- Base ----
FROM python:3.10-slim

# system deps (compile wheels, unzip, etc.)
RUN apt-get update && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ---- Python deps ----
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ---- Code & model ----
COPY app ./app
COPY car_price_model_tradein.pkl CarFaxCleanedData.csv ./

# ---- Run ----
ENV PYTHONUNBUFFERED=1
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]
