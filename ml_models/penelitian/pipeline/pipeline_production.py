from typing import Dict, Any
import numpy as np
import os
from .utils import (
    load_production_config, 
    load_pickle, 
    resolve_resource_path 
)
from .prepare_production_input import prepare_production_input

CONFIG = load_production_config()

def run_production(raw_input: Dict[str, Any]) -> Dict[str, Any]:
    print("\n[DEBUG] ================== run_production DIPANGGIL ==================")
    print("[DEBUG] raw_input yang diterima endpoint =", raw_input)

    # 1. Load Model
    model_path_cfg = CONFIG.get("model_path", "production_model/xgb_production.pkl")
    print(f"[DEBUG] model_path dari CONFIG = {model_path_cfg}")

    # ✅ path yang benar-benar dipakai backend (setelah resolve)
    model_path = resolve_resource_path(model_path_cfg)
    print(f"[DEBUG] model_path RESOLVED (abspath) = {os.path.abspath(model_path)}")

    model = load_pickle(model_path)
    print("[DEBUG] Model XGB berhasil di-load.")


    # 2. LOAD SCALER & YEO-JOHNSON
    mm_paths_to_try = [    
        "production_model/preprocessing/scaler_production.pkl",
    ]

    yj_paths_to_try = [                        
        "production_model/preprocessing/yj_production.pkl",
    ]

    mm_scaler = None
    yj_transformer = None

    # Loop cari file MinMax
    for path in mm_paths_to_try:
        full_path = resolve_resource_path(path)
        print(f"[DEBUG] Cek keberadaan MinMax di: {full_path}")
        if os.path.exists(full_path):
            print(f"[INFO] Production Pipeline: Menggunakan MinMax Scaler dari '{path}'")
            mm_scaler = load_pickle(full_path)
            break

    # Loop cari file Yeo-Johnson
    for path in yj_paths_to_try:
        full_path = resolve_resource_path(path)
        print(f"[DEBUG] Cek keberadaan YJ di: {full_path}")
        if os.path.exists(full_path):
            print(f"[INFO] Production Pipeline: Menggunakan YJ Transformer dari '{path}'")
            yj_transformer = load_pickle(full_path)
            break

    if mm_scaler is None:
        raise FileNotFoundError(
            "Gagal menemukan MinMaxScaler untuk target. "
            "Pastikan mm_target.pkl atau scaler_production.pkl ada di folder production_model."
        )

    if yj_transformer is None:
        raise FileNotFoundError(
            "Gagal menemukan Yeo-Johnson transformer untuk target. "
            "Pastikan yj_target.pkl atau yj_production.pkl ada di folder production_model."
        )

    if hasattr(mm_scaler, "feature_names_in_"):
        print("[DEBUG] mm_scaler.feature_names_in_ (len =", len(mm_scaler.feature_names_in_), ") =",
              list(mm_scaler.feature_names_in_))

    # --- B. PREPARE INPUT ---
    x, imputed_row = prepare_production_input(raw_input)

    # ===================================================
    # ✅ DEBUG: PRINT NILAI SEBELUM MASUK KE MODEL
    # ===================================================
    print("\n[DEBUG] ================== INPUT KE MODEL ==================")
    print("[DEBUG] x.shape =", x.shape)

    # Print semua nilai fitur yang masuk ke model
    if x.ndim == 2 and x.shape[0] > 0:
        print("[DEBUG] x[0] (nilai fitur berurutan):")
        for i, val in enumerate(x[0]):
            print(f"  [{i:02d}] {val}")
    else:
        print("[DEBUG] x array tidak valid:", x)

    # Print data imputasi (yang akan masuk DB)
    print("[DEBUG] imputed_row (data yang ditulis ke DB) =", imputed_row)

    # =============================
    # ✅ TAMBAHAN DEBUG NAMA FITUR
    # =============================
    try:
        from pathlib import Path
        import json

        feature_order_path = Path(__file__).resolve().parent.parent / "production_model" / "feature_order_production.json"

        if feature_order_path.exists():
            with open(feature_order_path, "r", encoding="utf-8") as f:
                feature_order = json.load(f)

            if len(feature_order) == x.shape[1]:
                print("\n[DEBUG] Feature Order (nama kolom input model):")
                for i, name in enumerate(feature_order):
                    print(f"  [{i:02d}] {name}")
            else:
                print("[DEBUG] WARNING: Panjang feature_order TIDAK sama dengan jumlah kolom X")
                print("[DEBUG] len(feature_order) =", len(feature_order))
                print("[DEBUG] x.shape[1]       =", x.shape[1])
        else:
            print("[DEBUG] feature_order_production.json tidak ditemukan:", feature_order_path)

    except Exception as e:
        print("[DEBUG] Gagal print feature_order:", e)

    print("[DEBUG] ================== END INPUT MODEL ==================\n")

    # --- C. PREDIKSI & INVERSE TRANSFORM ---
    y_pred = model.predict(x)

    y_scaled = float(y_pred[0]) if isinstance(y_pred, (list, tuple, np.ndarray)) else float(y_pred)

    print(f"[DEBUG] y_pred (raw dari model.predict) = {y_pred}")
    print(f"[DEBUG] y_scaled (scalar) = {y_scaled}")

    # --- helper: cari index kolom target di scaler/YJ ---
    target_candidates = [
        "Produksi (Ton)_yeojohnson",
        "produksi (ton)_yeojohnson",
        "hasil produksi (ton)_yeojohnson",
        "produksi (ton)",
        "hasil produksi (ton)",
        "Produksi (ton)_yeojohnson",
        "Hasil Produksi (ton)_yeojohnson",
        "Produksi (ton)",
        "Hasil Produksi (ton)",
    ]

    def find_target_index(obj, default_idx: int = -1) -> int:
        idx = default_idx
        if hasattr(obj, "feature_names_in_"):
            names = list(obj.feature_names_in_)
            print("[DEBUG] find_target_index: feature_names_in_ =", names)
            for cand in target_candidates:
                if cand in names:
                    print(f"[DEBUG] find_target_index: pakai cand '{cand}' sebagai target")
                    idx = names.index(cand)
                    break
        else:
            print("[DEBUG] find_target_index: tidak ada feature_names_in_ pada objek", type(obj))
        return idx

    # --- 2) Inverse MinMax: 0–1 -> ruang Yeo-Johnson (z_std) ---
    if hasattr(mm_scaler, "n_features_in_") and mm_scaler.n_features_in_ > 1:
        n_feats_mm = mm_scaler.n_features_in_
        target_idx_mm = find_target_index(mm_scaler, default_idx=n_feats_mm - 1)

        print(f"[DEBUG] n_feats_mm (jumlah fitur di MinMax) = {n_feats_mm}")
        print(f"[DEBUG] target_idx_mm (index target di MinMax) = {target_idx_mm}")
        if hasattr(mm_scaler, "feature_names_in_") and 0 <= target_idx_mm < len(mm_scaler.feature_names_in_):
            print("[DEBUG] Nama kolom target di MinMax =", mm_scaler.feature_names_in_[target_idx_mm])

        if target_idx_mm < 0 or target_idx_mm >= n_feats_mm:
            raise ValueError(
                f"Tidak dapat menentukan index target di MinMaxScaler (n_features={n_feats_mm})."
            )

        y_scaled_full = np.zeros((1, n_feats_mm), dtype=float)
        y_scaled_full[0, target_idx_mm] = y_scaled

        print("[DEBUG] y_scaled_full (vektor input ke inverse MinMax) =", y_scaled_full)

        y_yj_full = mm_scaler.inverse_transform(y_scaled_full)
        y_yj_val = float(y_yj_full[0, target_idx_mm])

        print("[DEBUG] y_yj_full (hasil inverse MinMax) =", y_yj_full)
        print("[DEBUG] y_yj_val (elemen target setelah inverse MinMax) =", y_yj_val)

    else:
        print("[DEBUG] MinMax scaler hanya 1 kolom atau tidak punya n_features_in_, pakai mode 1 kolom.")
        y_scaled_arr = np.array([[y_scaled]], dtype=float)
        y_yj_arr = mm_scaler.inverse_transform(y_scaled_arr)
        y_yj_val = float(y_yj_arr[0, 0])
        print("[DEBUG] y_yj_arr (1 kolom) =", y_yj_arr)
        print("[DEBUG] y_yj_val (1 kolom) =", y_yj_val)

    # --- helper: pilih objek YJ yang benar kalau yj_transformer berupa dict ---
    def select_yj_object(yj_obj):
        """
        Jika yj_obj adalah dict (berisi beberapa transformer),
        pilih satu yang dipakai untuk target.
        """
        if not isinstance(yj_obj, dict):
            print("[DEBUG] yj_transformer bukan dict, langsung pakai objeknya.")
            return yj_obj

        print("[DEBUG] yj_transformer adalah dict, keys =", list(yj_obj.keys()))
        priority_words = ["target", "produksi", "production", "y"]
        for key, val in yj_obj.items():
            if not hasattr(val, "inverse_transform"):
                continue
            lk = str(key).lower()
            if any(w in lk for w in priority_words):
                print(f"[INFO] YJ: memakai transformer dari key '{key}' (priority match)")
                return val

        for key, val in yj_obj.items():
            if hasattr(val, "inverse_transform") and hasattr(val, "n_features_in_"):
                try:
                    if int(val.n_features_in_) == 1:
                        print(f"[INFO] YJ: memakai transformer (1 kolom) dari key '{key}'")
                        return val
                except Exception:
                    pass

        for key, val in yj_obj.items():
            if hasattr(val, "inverse_transform"):
                print(f"[INFO] YJ: memakai transformer pertama dari key '{key}' (fallback)")
                return val

        raise ValueError(
            "Tidak dapat menemukan objek Yeo-Johnson yang punya inverse_transform di dalam yj_production.pkl"
        )

    # --- 3) Inverse Yeo-Johnson: ruang YJ -> ton asli ---
    yj_obj = select_yj_object(yj_transformer)

    if hasattr(yj_obj, "n_features_in_") and yj_obj.n_features_in_ > 1:
        n_feats_yj = yj_obj.n_features_in_
        target_idx_yj = find_target_index(yj_obj, default_idx=n_feats_yj - 1)

        print(f"[DEBUG] n_feats_yj (jumlah fitur di YJ) = {n_feats_yj}")
        print(f"[DEBUG] target_idx_yj (index target di YJ) = {target_idx_yj}")
        if hasattr(yj_obj, "feature_names_in_") and 0 <= target_idx_yj < len(yj_obj.feature_names_in_):
            print("[DEBUG] Nama kolom target di YJ =", yj_obj.feature_names_in_[target_idx_yj])

        if target_idx_yj < 0 or target_idx_yj >= n_feats_yj:
            raise ValueError(
                f"Tidak dapat menentukan index target di Yeo-Johnson transformer (n_features={n_feats_yj})."
            )

        y_yj_full_in = np.zeros((1, n_feats_yj), dtype=float)
        y_yj_full_in[0, target_idx_yj] = y_yj_val

        print("[DEBUG] y_yj_full_in (vektor input ke inverse YJ) =", y_yj_full_in)

        y_final_full = yj_obj.inverse_transform(y_yj_full_in)
        y_final_ton = float(y_final_full[0, target_idx_yj])

        print("[DEBUG] y_final_full (hasil inverse YJ) =", y_final_full)
    else:
        print("[DEBUG] YJ transformer 1 kolom atau tidak punya n_features_in_, pakai mode 1 kolom.")
        y_yj_arr = np.array([[y_yj_val]], dtype=float)
        y_final_arr = yj_obj.inverse_transform(y_yj_arr)
        y_final_ton = float(y_final_arr[0, 0])
        print("[DEBUG] y_final_arr (1 kolom) =", y_final_arr)

    # --- LOG DEBUG UTAMA ---
    print(f"Prediksi Raw (0-1) dari model       : {y_scaled}")
    print(f"Prediksi YJ (setelah inverse MinMax): {y_yj_val}")
    print(f"Prediksi Final (Ton, sebelum x0.6204): {y_final_ton}")

    output_field = CONFIG.get("output_schema", {}).get("field_name", "prediksi_produksi")
    print("[DEBUG] output_field =", output_field)
    print("[DEBUG] imputed_features['Produktivitas (ton/ha)'] =",
          imputed_row.get("Produktivitas (ton/ha)", None))

    print("[DEBUG] ================== Selesai run_production ==================\n")

    return {
        output_field: y_final_ton,
        "imputed_features": imputed_row,
    }
