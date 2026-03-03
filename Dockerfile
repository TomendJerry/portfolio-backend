# Gunakan image Python yang sangat ringan
FROM python:3.10-slim

WORKDIR /app

# Hanya instal library yang benar-benar dibutuhkan untuk psycopg2-binary
# libpq-dev tetap dibutuhkan untuk runtime PostgreSQL
RUN apt-get update && apt-get install -y libpq-dev && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# Upgrade pip dan instal dependensi
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY . .

# Render menggunakan port 10000 secara default untuk paket gratis
EXPOSE 10000

# Pastikan port sesuai dengan EXPOSE dan host diatur ke 0.0.0.0
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "10000"]