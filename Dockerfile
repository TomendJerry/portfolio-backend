# Gunakan image Python yang ringan
FROM python:3.10-slim

WORKDIR /app

# Instal dependensi sistem yang diperlukan untuk PostgreSQL
RUN apt-get update && apt-get install -y libpq-dev gcc && rm -rf /var/lib/apt/lists/*

# Copy dan instal requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy seluruh kode backend
COPY . .

# Ekspos port yang digunakan di main.py
EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]