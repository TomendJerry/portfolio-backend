from typing import Dict, Any
import os
import numpy as np
import pandas as pd
import json
from .utils import (
    load_production_config,
    load_pickle,
    load_province_mapping,
    resolve_province_key,
    get_imputation_folder,
    apply_gain_imputation,
    ensure_columns_order,
)

CONFIG = load_production_config()
PROVINCE_MAPPING = load_province_mapping()

_PREPROC_CFG = CONFIG.get("preprocessing", {})
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 1. Ambil path dari config (boleh relative), kalau tidak ada pakai default
_FEATURE_ORDER_PATH = CONFIG.get(
    "feature_order_path",
    os.path.join("production_model", "feature_order_production.json")
)

# 2. Jika path belum absolut → buat relatif ke folder ml (.. dari pipeline)
if not os.path.isabs(_FEATURE_ORDER_PATH):
    _FEATURE_ORDER_PATH = os.path.join(
        BASE_DIR,
        "..",               # dari backend/ml/pipeline → backend/ml
        _FEATURE_ORDER_PATH # "production_model/feature_order_production.json"
    )

# 3. Normalisasi dan load file JSON
_FEATURE_ORDER_PATH = os.path.normpath(_FEATURE_ORDER_PATH)
with open(_FEATURE_ORDER_PATH, "r", encoding="utf-8") as f:
    feature_order = json.load(f)

# DAFTAR FITUR NUMERIK (Versi Lowercase, sesuai Training)
NUMERIC_FEATURES_LOWER = [
    "luas panen (ha)",
    "produktivitas",
    "suhu min (oc)",
    "suhu rata-rata (oc)",
    "suhu maks (oc)",
    "kelembapan (%)",
    "curah hujan (mm)",
    "jumlah hari hujan (hari)",
    "kecepatan angin (m/s)",
    "tekanan udara (mb)",
    "penyinaran matahari (%)",
    "annual_soi_bom",
    "annual_dmi",
]

def prepare_production_input(raw_input: Dict[str, Any]) -> np.ndarray:
    print("\n[DEBUG] ================== Mulai Prepare Input ==================")
    print("[DEBUG] raw_input (dari frontend) =", raw_input)

    # 1. Normalisasi Input ke Lowercase (Agar 100% sama dengan Training)
    # ------------------------------------------------------------------
    input_lower = {}
    for k, v in raw_input.items():
        # Bersihkan key: trim spasi, lowercase
        clean_k = k.strip().lower()
        input_lower[clean_k] = v
    
    # Debug: Cek apakah data Luas Panen ada?
    print(f"[DEBUG] Luas Panen (raw, lower key): {input_lower.get('luas panen (ha)')}")
    print(f"[DEBUG] Tahun (raw, lower key): {input_lower.get('tahun')}")

    # 2. Validasi Provinsi
    province_raw = input_lower.get("provinsi")
    if not province_raw:
        raise ValueError("Field 'Provinsi' wajib diisi.")
    
    # province_key = key di province_mapping.json (misal: "Nanggroe Aceh Darussalam")
    province_key = resolve_province_key(province_raw, PROVINCE_MAPPING)

    # 🔑 NILAI UNTUK OHE HARUS SAMA SEPERTI SAAT TRAINING → lowercase
    province_ohe_value = province_key.strip().lower()
    print(f"[DEBUG] province_raw = {province_raw} | province_key = {province_key} | province_ohe_value = {province_ohe_value}")

    # 3. Imputasi GAIN
    imputation_cfg = CONFIG.get("imputation", {})
    base_path = imputation_cfg.get("base_path", "production_model/imputation")
    province_folder = get_imputation_folder(province_key, PROVINCE_MAPPING)

    df_raw = pd.DataFrame([input_lower])
    df_raw = df_raw.replace(["", " "], np.nan)

    print("[DEBUG] df_raw (1 baris, sebelum GAIN) =", df_raw.iloc[0].to_dict())
    print(f"[DEBUG] apply_gain_imputation base_path={base_path}, province_folder={province_folder}")

    # Apply GAIN (Output GAIN biasanya lowercase jika trainingnya lowercase)
    df_gain = apply_gain_imputation(df_raw, base_path, province_folder)
    
    # Pastikan df_gain juga lowercase kolomnya
    df_gain.columns = df_gain.columns.str.strip().str.lower()

    # Merge Input User vs Imputasi
    cols_to_use = df_gain.columns.intersection(df_raw.columns)
    df_imputed = df_gain.copy()
    df_imputed[cols_to_use] = df_raw[cols_to_use].where(
        df_raw[cols_to_use].notna(), 
        df_gain[cols_to_use]
    )

    print("[DEBUG] df_imputed (1 baris, setelah GAIN+merge) =", df_imputed.iloc[0].to_dict())

    # 4. Ambil Fitur Numerik (Pasti Lowercase)
    # Pastikan semua kolom numerik ada, isi 0.0 jika tidak
    for col in NUMERIC_FEATURES_LOWER:
        if col not in df_imputed.columns:
            print(f"[DEBUG] Warning: Kolom '{col}' tidak ditemukan di df_imputed, set 0.0")
            df_imputed[col] = 0.0

    # 🔍 DEBUG 1: cek produktivitas setelah imputasi GAIN / input user
    if "produktivitas" in df_imputed.columns:
        print("[DEBUG] df_imputed['produktivitas'] =", df_imputed["produktivitas"].iloc[0])
    else:
        print("[DEBUG] df_imputed TIDAK punya kolom 'produktivitas'!!")
                
    num_df = df_imputed[NUMERIC_FEATURES_LOWER].copy()
    print(f"[DEBUG] Sampel Numerik sebelum YJ (num_df) =", num_df.iloc[0].to_dict())

    # 5. OHE Provinsi
    cat_df = pd.DataFrame({"provinsi": [province_ohe_value]})
    
    ohe = load_pickle(_PREPROC_CFG["ohe_path"]) if _PREPROC_CFG.get("ohe_path") else None
    yj = load_pickle(_PREPROC_CFG["yeo_johnson_path"]) if _PREPROC_CFG.get("yeo_johnson_path") else None
    scaler = load_pickle(_PREPROC_CFG["scaler_path"]) if _PREPROC_CFG.get("scaler_path") else None

    if ohe:
        try:
            print("[DEBUG] OHE: transform 'provinsi' dengan encoder dari", _PREPROC_CFG.get("ohe_path"))
            ohe_arr = ohe.transform(cat_df)
            ohe_cols = ohe.get_feature_names_out(cat_df.columns)
            ohe_df = pd.DataFrame(ohe_arr, columns=ohe_cols)
            ohe_df.columns = [c.strip().lower() for c in ohe_df.columns]
            print("[DEBUG] OHE Provinsi columns =", list(ohe_df.columns))
            print("[DEBUG] OHE Provinsi row[0] =", ohe_df.iloc[0].to_dict())
        except Exception as e:
            print(f"[DEBUG] OHE Error (mungkin beda nama kolom): {e}")
            ohe_feats = list(ohe.get_feature_names_out())
            ohe_df = pd.DataFrame(0, index=[0], columns=ohe_feats)
    else:
        print("[DEBUG] OHE Provinsi TIDAK dimuat (ohe_path kosong atau file tidak ada).")
        ohe_df = pd.DataFrame()

    # 6. Gabung Numerik + OHE
    merged = pd.concat([num_df.reset_index(drop=True), ohe_df.reset_index(drop=True)], axis=1)
    print("[DEBUG] merged (num_df + OHE) columns =", list(merged.columns))
    print("[DEBUG] merged row[0] =", merged.iloc[0].to_dict())

    # 7. Transformasi Yeo-Johnson (PERBAIKAN LOGIC)
    if yj:
        print("[DEBUG] YJ Transformer ditemukan, tipe =", type(yj))
        if isinstance(yj, dict):
            yj_data = {}
            for col in merged.columns:
                transformer = None
                
                # --- LOGIKA BARU: Coba variasi nama key ---
                candidates = [
                    col,                                # 1. Nama asli
                    col.lower(),                        # 2. Lowercase
                    f"{col}_yeojohnson",                # 3. Nama asli + suffix
                    f"{col.lower()}_yeojohnson"         # 4. Lowercase + suffix (INI YANG BENAR!)
                ]
                
                for key in candidates:
                    if key in yj:
                        transformer = yj[key]
                        # Debug key yang terpakai
                        print(f"[DEBUG] YJ: pakai key '{key}' untuk kolom '{col}'")
                        break
                
                if transformer:
                    try:
                        vals = merged[[col]].values
                        trans_vals = transformer.transform(vals)
                        yj_data[col] = trans_vals.ravel()
                    except Exception as e:
                        print(f"[DEBUG] Gagal transform YJ untuk {col}: {e}")
                        yj_data[col] = merged[col].values
                else:
                    # tidak ada transformer untuk kolom ini
                    # print(f"[DEBUG] No YJ transformer for {col}, passing raw.")
                    yj_data[col] = merged[col].values
            
            yj_df = pd.DataFrame(yj_data, index=merged.index)
            yj_df = yj_df[merged.columns]
        else:
            print("[DEBUG] YJ berupa single transformer, apply ke seluruh merged.")
            yj_arr = yj.transform(merged)
            yj_df = pd.DataFrame(yj_arr, columns=merged.columns)
    else:
        print("[DEBUG] TIDAK ada YJ transformer, pakai merged apa adanya.")
        yj_df = merged.copy()

    # 8. Rename Kolom untuk Scaler (Tambah _yeojohnson)
    rename_map = {}
    for col in yj_df.columns:
        if col in NUMERIC_FEATURES_LOWER:
            rename_map[col] = f"{col}_yeojohnson"
        else:
            rename_map[col] = col # OHE biarkan
    
    yj_df = yj_df.rename(columns=rename_map)
    
    # 🔍 DEBUG 2: cek produktivitas_yeojohnson sebelum scaling
    if "produktivitas_yeojohnson" in yj_df.columns:
        print("[DEBUG] yj_df['produktivitas_yeojohnson'] =", yj_df["produktivitas_yeojohnson"].iloc[0])
    else:
        print("[DEBUG] yj_df TIDAK punya kolom 'produktivitas_yeojohnson'!!")
    
    print("[DEBUG] yj_df columns (setelah rename) =", list(yj_df.columns))

    # 8b. Bentuk label SOI_BOM (one-hot: el nino / la nina / netral)
    soi_val = None
    if "annual_soi_bom" in df_imputed.columns:
        try:
            soi_val = float(df_imputed["annual_soi_bom"].iloc[0])
        except (TypeError, ValueError):
            soi_val = None

    yj_df["soi_bom_label_el nino"] = 0.0
    yj_df["soi_bom_label_la nina"] = 0.0
    yj_df["soi_bom_label_netral"] = 0.0

    if soi_val is not None:
        if soi_val > 7:
            yj_df["soi_bom_label_la nina"] = 1.0
            soi_cat = "la nina"
        elif soi_val < -7:
            yj_df["soi_bom_label_el nino"] = 1.0
            soi_cat = "el nino"
        else:
            yj_df["soi_bom_label_netral"] = 1.0
            soi_cat = "netral"
    else:
        soi_cat = "unknown"

    print(f"[DEBUG] annual_soi_bom = {soi_val}, diklasifikasikan sebagai '{soi_cat}'")
    print("[DEBUG] Kolom SOI one-hot:",
          yj_df[["soi_bom_label_el nino", "soi_bom_label_la nina", "soi_bom_label_netral"]].iloc[0].to_dict())

    # 9. Tambah 'tahun' jika diperlukan
    if "tahun" in feature_order and "tahun" not in yj_df.columns:
        yj_df["tahun"] = input_lower.get("tahun", 2026)

    # ============================================
    # 10. Siapkan fitur numerik (scaling MinMax)
    # ============================================
    numeric_bases = NUMERIC_FEATURES_LOWER + ["produksi (ton)"]

    X_for_scaler = pd.DataFrame(index=yj_df.index)

    if scaler is not None:
        print("[DEBUG] Scaler feature_names_in_ =", getattr(scaler, "feature_names_in_", None))
        scaler_feature_names = list(getattr(scaler, "feature_names_in_", []))
        if not scaler_feature_names:
            scaler_feature_names = numeric_bases

        for base in scaler_feature_names:
            if base == "produksi (ton)":
                X_for_scaler[base] = float(input_lower.get("produksi (ton)", 0.0))
                continue

            suff = f"{base}_yeojohnson"
            if suff in yj_df.columns:
                X_for_scaler[base] = yj_df[suff]
            elif base in yj_df.columns:
                X_for_scaler[base] = yj_df[base]
            else:
                X_for_scaler[base] = 0.0

        X_for_scaler = X_for_scaler[scaler_feature_names]

        print("[DEBUG] X_for_scaler columns (urutan ke scaler) =", list(X_for_scaler.columns))
        print("[DEBUG] X_for_scaler row[0] =", X_for_scaler.iloc[0].to_dict())

        scaled_arr = scaler.transform(X_for_scaler)
        X_num_scaled = pd.DataFrame(
            scaled_arr,
            columns=scaler_feature_names,
            index=yj_df.index,
        )
        print("[DEBUG] X_num_scaled row[0] =", X_num_scaled.iloc[0].to_dict())

    else:
        print("[DEBUG] TIDAK ada scaler, pakai nilai YJ/raw langsung.")
        for base in numeric_bases:
            suff = f"{base}_yeojohnson"
            if suff in yj_df.columns:
                X_for_scaler[base] = yj_df[suff]
            elif base in yj_df.columns:
                X_for_scaler[base] = yj_df[base]
            else:
                X_for_scaler[base] = 0.0

        X_num_scaled = X_for_scaler.copy()

    # ============================================
    # 11. Susun semua fitur sesuai feature_order
    # ============================================
    combined = pd.DataFrame(index=yj_df.index)

    print(f"[DEBUG] Panjang feature_order = {len(feature_order)}")
    # print("[DEBUG] feature_order =", feature_order)  # aktifkan jika perlu

    for col in feature_order:
        col_lower = col.strip().lower()

        if col == "tahun":
            if "tahun" in yj_df.columns:
                combined[col] = yj_df["tahun"].values
            else:
                combined[col] = np.array(
                    [input_lower.get("tahun", 2026)],
                    dtype="float32",
                )

        elif col_lower in numeric_bases:
            if col_lower in X_num_scaled.columns:
                combined[col] = X_num_scaled[col_lower].values
            else:
                combined[col] = np.zeros(len(yj_df), dtype="float32")

        elif col.startswith("provinsi_") or col.startswith("soi_bom_label_"):
            if col in yj_df.columns:
                combined[col] = yj_df[col].values
            else:
                combined[col] = np.zeros(len(yj_df), dtype="float32")

        else:
            combined[col] = np.zeros(len(yj_df), dtype="float32")

    combined = combined[feature_order]

    print("[DEBUG] combined (final sebelum to_numpy) columns =", list(combined.columns)[:15], "...")
    print("[DEBUG] combined row[0] (subset 15 kolom pertama) =", combined.iloc[0, :15].to_dict())

    # ============================================
    # 12. Kembalikan sebagai numpy array
    # ============================================
    X_array = combined.to_numpy().astype("float32")

    # =====================================================
    # 11. Siapkan imputed_features untuk ditulis ke DB
    # =====================================================
    row = df_imputed.iloc[0]

    db_cols_map = {
        "Luas Panen (ha)": "luas panen (ha)",
        "Suhu Min (oC)": "suhu min (oc)",
        "Suhu Rata-rata (oC)": "suhu rata-rata (oc)",
        "Suhu Maks (oC)": "suhu maks (oc)",
        "Kelembapan (%)": "kelembapan (%)",
        "Curah Hujan (mm)": "curah hujan (mm)",
        "Jumlah Hari Hujan (hari)": "jumlah hari hujan (hari)",
        "Kecepatan Angin (m/s)": "kecepatan angin (m/s)",
        "Tekanan Udara (mb)": "tekanan udara (mb)",
        "Penyinaran Matahari (%)": "penyinaran matahari (%)",
        "Annual_SOI_BOM": "annual_soi_bom",
        "Annual_DMI": "annual_dmi",
        "Produktivitas (ton/ha)": "produktivitas",
    }

    imputed_features: dict[str, float | None] = {}
    for db_name, lower_name in db_cols_map.items():
        if lower_name in df_imputed.columns:
            val = row[lower_name]
            if pd.isna(val) or val is None or val == "":
                imputed_features[db_name] = None
            else:
                try:
                    imputed_features[db_name] = float(val)
                except (TypeError, ValueError):
                    imputed_features[db_name] = None
        else:
            imputed_features[db_name] = None

    print("[DEBUG] X_array shape =", X_array.shape)
    if X_array.shape[1] > 0:
        print("[DEBUG] X_array[0][:15] =", X_array[0][:15])
    print("[DEBUG] imputed_features =", imputed_features)
    print("[DEBUG] ================== Selesai Prepare Input ==================\n")

    return X_array, imputed_features
