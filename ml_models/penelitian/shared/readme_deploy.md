# Deploy Guide — Rice Production & Demand Prediction System

This document explains how to run the deployment pipeline for:

- **Model 1: Prediksi Produksi (XGBoost)**
- **Model 2: Prediksi Permintaan (Hybrid BiLSTM-BiGRU)**
- **GAIN Imputation per provinsi**
- **Complete preprocessing pipeline**

Folder utama: `deploy_ready/`  
Struktur folder lengkap sudah disesuaikan untuk production deployment.

---

# 1. Struktur Folder

```
deploy_ready/
│
├── config/
│   ├── production_config.json
│   └── demand_config.json
│
├── production_model/
│   ├── xgb_production.pkl
│   ├── feature_order_production.json
│   ├── preprocessing/
│   └── imputation/ (per provinsi)
│
├── demand_model/
│   ├── hybrid_demand.h5
│   ├── feature_order_demand.json
│   ├── preprocessing/
│   └── imputation/ (per provinsi)
│
├── pipeline/
│   ├── prepare_production_input.py
│   ├── prepare_demand_input.py
│   ├── pipeline_production.py
│   ├── pipeline_demand.py
│   └── utils.py
│
├── data/
│   └── historical_master_demand.xlsx
│
└── shared/
    ├── province_mapping.json
    ├── example_input_production.json
    ├── example_input_demand.json
    └── readme_deploy.md
```

---

# 2. Cara Load Configuration

Setiap model memiliki config masing-masing:

## Model Produksi
```
config/production_config.json
```

## Model Permintaan
```
config/demand_config.json
```

Untuk load config:

```python
import json

with open("config/production_config.json") as f:
    config_prod = json.load(f)

with open("config/demand_config.json") as f:
    config_dem = json.load(f)
```

---

# 3. Pipeline Produksi

## 3.1. Input Required
Contoh input untuk API model produksi:

```
shared/example_input_production.json
```

Pipeline akan:
1. Imputasi GAIN (per provinsi)
2. Apply OHE → Yeo-Johnson → MinMax Scaler
3. Prediksi produksi pakai XGBoost

## 3.2. Jalankan pipeline
```python
from pipeline.pipeline_production import run_production

output = run_production(input_data)
print(output)
```

Output field:
```
prediksi_produksi (ton)
```

---

# 4. Pipeline Permintaan (Hybrid)

Model permintaan membutuhkan:
- Input mentah
- **Prediksi produksi dari model 1**
- Sequence 12 tahun ke belakang (diambil dari `data/historical_master_demand.xlsx`)

## 4.1. Jalankan pipeline
```python
from pipeline.pipeline_demand import run_demand

output = run_demand(input_data)
print(output)
```

Output field:
```
prediksi_permintaan (ton)
```

---

# 5. GAIN Imputation

GAIN disimpan per provinsi:

```
/production_model/imputation/Aceh/
    gain_generator.h5
    scaler_gain.pkl
    col_means.npy
    feature_names.json
```

GAIN akan otomatis dipanggil berdasarkan:

```python
provinsi = input_data["provinsi"]
```

---

# 6. Preprocessing Rules

### Produksi:
- OHE: `ohe_province_production.pkl`
- Yeo-Johnson: `yj_production.pkl`
- Scaler: `scaler_production.pkl`

### Permintaan:
- OHE: `ohe_province_demand.pkl`
- Yeo-Johnson: `yj_demand.pkl`
- Scaler: `scaler_demand.pkl`

---

# 7. Dependency Handling

Model permintaan akan otomatis memanggil:

```python
pipeline/pipeline_production.py
```

untuk mendapatkan:
```
prediksi_produksi
```

---

# 8. Catatan untuk Developer

- **Jangan training ulang model** di server — semua file sudah final.
- Pastikan environment sudah mengandung:
  - TensorFlow
  - XGBoost
  - Scikit-learn
  - Numpy
  - Pandas
- Path di config relatif → aman dipindah folder.

---

# 9. Troubleshooting

| Masalah | Penyebab | Solusi |
|--------|----------|--------|
| Bentuk input salah | Tidak mengikuti feature order | Lihat `feature_order_*.json` |
| Error GAIN | Folder provinsi tidak ditemukan | Pastikan nama provinsi sama |
| Output None | Sequence history kurang | Cek `historical_master_demand.xlsx` |

---
