from typing import Dict, Any
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from .utils import (
    load_demand_config,
    load_province_mapping,
    resolve_province_key,
    get_imputation_folder,
    apply_gain_imputation,
    load_pickle,
    ensure_columns_order,
    resolve_resource_path,
)

CONFIG = load_demand_config()
PROVINCE_MAPPING = load_province_mapping()

# Daftar 12 Fitur Numerik sesuai arsitektur model Demand
NUMERIC_FEATURES = [
    "Jumlah Penduduk (Jiwa)",
    "Laju Pertumbuhan Penduduk (%)",
    "Harga Beras (Rp/Kg)",
    "Harga Jagung (Rp/Kg)",
    "Harga Mie Instan (Rp/Bungkus)",
    "Tingkat Kemiskinan (%)",
    "Garis Kemiskinan per Kapita (Rp/tahun)",
    "Produk Domestik Regional Bruto per Kapita HK (Ribu Rp)",
    "Gini Rasio",
    "Rata-rata Banyaknya Anggota Rumah Tangga (Jiwa)",
    "Tingkat Pengangguran Terbuka (%)",
    "Produksi (ton)"
]

def build_sequence_with_history(raw_input: dict, produksi_prediksi: float, seq_len: int = 12) -> pd.DataFrame:
    """
    Membangun urutan data (sequence) gabungan antara:
    1. Data Historis (dari DB via inject, atau Excel via config)
    2. Data Prediksi saat ini (Input User)
    """
    
    # 1. CEK DATA HISTORIS DARI DB (INJECTED via Router)
    if "__historical_df_override" in raw_input and raw_input["__historical_df_override"] is not None:
        df_hist = raw_input["__historical_df_override"]
        if "Provinsi" not in df_hist.columns and not df_hist.empty:
             df_hist["Provinsi"] = raw_input.get("Provinsi", "Unknown")
    else:
        data_cfg = CONFIG.get("data_required", {})
        hist_path = data_cfg.get("historical_demand_data")
        if not hist_path:
            raise ValueError("Path historical_demand_data belum di-set")
        hist_path = resolve_resource_path(hist_path)
        df_hist = pd.read_excel(hist_path)

    province_raw = raw_input.get("Provinsi") or raw_input.get("provinsi")
    province_key = resolve_province_key(province_raw, PROVINCE_MAPPING)
    
    if "__historical_df_override" in raw_input:
        df_p = df_hist.copy()
    else:
        dataset_label = PROVINCE_MAPPING[province_key].get("dataset_label", province_key)
        df_p = df_hist[df_hist["Provinsi"] == dataset_label].copy()

    df_p = df_p.sort_values("Tahun")

    # Padding jika data kurang
    if len(df_p) < seq_len - 1:
        missing_count = (seq_len - 1) - len(df_p)
        if missing_count > 0 and len(df_p) > 0:
            first_row = df_p.iloc[0:1]
            df_p = pd.concat([first_row] * missing_count + [df_p], ignore_index=True)
        elif len(df_p) == 0:
             # Buat dummy frame kosong jika DB kosong sama sekali (untuk hindari crash total)
             # Tapi idealnya ini harusnya error karena LSTM butuh history.
             raise ValueError(f"Data historis kosong untuk {province_key}.")

    df_last = df_p.tail(seq_len - 1).copy()

    # --- PERUBAHAN PENTING DI SINI ---
    # Buat baris baru. JANGAN sekadar copy baris terakhir, 
    # karena kita ingin GAIN memprediksi perubahan, bukan menganggapnya statis.
    
    # 1. Ambil struktur kolom dari df_last
    new_row_data = {col: np.nan for col in df_last.columns} # Inisialisasi semua NaN
    
    # 2. Isi Tahun & Provinsi
    target_year = raw_input.get("tahun") or raw_input.get("Tahun") or (df_last.iloc[-1]["Tahun"] + 1)
    new_row_data["Tahun"] = target_year
    new_row_data["Provinsi"] = raw_input.get("Provinsi", df_last.iloc[-1]["Provinsi"])

    # 3. Isi Fitur dari User Input (Luas Panen, Curah Hujan, dll)
    for k, v in raw_input.items():
        if k == "__historical_df_override": continue
        # Cocokkan nama kolom secara persis
        if k in new_row_data:
            new_row_data[k] = v

    # 4. Isi Produksi (Hasil Prediksi XGBoost)
    # Cari nama kolom produksi yang valid di dataset history
    prod_col_candidates = ["Produksi (ton)", "produksi (ton)", "Produksi", "produksi"]
    target_prod_col = None
    for c in prod_col_candidates:
        if c in df_last.columns:
            target_prod_col = c
            break
            
    if target_prod_col:
        new_row_data[target_prod_col] = produksi_prediksi

    # 5. Buat DataFrame 1 baris
    new_row = pd.DataFrame([new_row_data])

    # Gabungkan: History + New Row (yang banyak NaN-nya)
    # NaN ini nanti akan diisi oleh apply_gain_imputation di langkah selanjutnya
    df_seq = pd.concat([df_last, new_row], axis=0, ignore_index=True)

    return df_seq

def prepare_demand_input(
    raw_input: Dict[str, Any],
    produksi_prediksi: float
) -> tuple[np.ndarray, dict]:
    
    # 1. Bangun Sequence Data
    seq_df = build_sequence_with_history(raw_input, produksi_prediksi)
    
    # Bersihkan nilai kosong agar siap untuk GAIN
    seq_df = seq_df.replace(["", " "], np.nan)
    
    # 2. Siapkan Imputasi GAIN
    province_raw = raw_input.get("Provinsi") or raw_input.get("provinsi")
    province_key = resolve_province_key(province_raw, PROVINCE_MAPPING)
    imputation_cfg = CONFIG.get("imputation", {})
    imputation_base = resolve_resource_path(
        imputation_cfg.get("base_path", "demand_model/imputation")
    )
    province_folder = get_imputation_folder(province_key, PROVINCE_MAPPING)

    # 3. Lakukan Imputasi GAIN
    seq_imputed = apply_gain_imputation(seq_df, imputation_base, province_folder)

    # 4. Simpan Log Fitur Hasil Imputasi (Untuk dikirim balik ke API/Frontend)
    last_row = seq_imputed.iloc[-1]
    
    # Mapping nama kolom agar seragam
    imputed_features: dict[str, float | None] = {}
    for col in NUMERIC_FEATURES:
        if col in seq_imputed.columns:
            val = last_row[col]
            try:
                if pd.isna(val):
                    imputed_features[col] = None
                else:
                    imputed_features[col] = float(val)
            except:
                imputed_features[col] = None

    # 5. Preprocessing Model (Yeo-Johnson & MinMax)
    prep_cfg = CONFIG.get("preprocessing", {})
    yj_path = resolve_resource_path(prep_cfg.get("yeo_johnson_path"))
    scaler_path = resolve_resource_path(prep_cfg.get("scaler_path"))

    yj = load_pickle(yj_path)
    scaler = load_pickle(scaler_path)

    # A. Ambil Fitur Numerik Saja
    features_df = seq_imputed[NUMERIC_FEATURES].copy().astype(float)

    # B. Transformasi Yeo-Johnson
    try:
        # Coba transform sekaligus
        yj_arr = yj.transform(features_df)
        yj_df = pd.DataFrame(yj_arr, columns=NUMERIC_FEATURES, index=features_df.index)
    except ValueError:
        # Fallback jika model dilatih per kolom
        yj_df = pd.DataFrame(index=features_df.index)
        for col in NUMERIC_FEATURES:
            yj_df[col] = yj.transform(features_df[[col]]).ravel()

    # C. Scaling MinMax
    scaled_arr = scaler.transform(yj_df)
    scaled_df = pd.DataFrame(scaled_arr, columns=NUMERIC_FEATURES, index=features_df.index)

    # D. Scaling Tahun (Manual 1993-2020 sesuai training notebook)
    year_min, year_max = 1993.0, 2020.0
    year_series = seq_imputed['Tahun'].astype(float)
    scaled_year = (year_series - year_min) / (year_max - year_min)

    # E. One-Hot Encoding Provinsi (Manual)
    # Model dilatih dengan One-Hot Encoding, kita harus membuatnya manual
    current_prov = seq_imputed['Provinsi'].iloc[0]
    # Buat DataFrame kosong dengan index yang sama
    ohe_df = pd.DataFrame(index=features_df.index)
    # Format nama kolom OHE harus SAMA PERSIS dengan training: "Provinsi_NamaProvinsi"
    ohe_df[f"Provinsi_{current_prov}"] = 1.0 

    # F. Gabungkan Semua Fitur
    final_df_pre = pd.concat([scaled_year.rename("Tahun"), scaled_df, ohe_df], axis=1)

    # G. Urutkan Kolom Sesuai Model
    # Fitur yang tidak ada di OHE (provinsi lain) akan diisi 0.0 otomatis oleh fungsi ini
    feature_order_path = CONFIG.get("feature_order_path", "demand_model/feature_order_demand.json")
    final_df = ensure_columns_order(final_df_pre, feature_order_path)

    # 6. Reshape ke format LSTM (Sample, TimeSteps, Features)
    seq_len = CONFIG.get("data_required", {}).get("sequence_length", 12)
    n_features = final_df.shape[1]
    
    arr = final_df.to_numpy().astype("float32")
    arr = arr.reshape(1, seq_len, n_features)
    
    return arr, imputed_features