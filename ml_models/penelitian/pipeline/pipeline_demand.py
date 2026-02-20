from typing import Dict, Any
import numpy as np
from tensorflow.keras.models import load_model

# utils.py berada di folder yang sama dengan pipeline_demand.py
from .utils import (
    load_demand_config,
    load_province_mapping,
    load_feature_order,
    load_historical_demand_data,
    load_imputation_components,
    _resolve_ml_path,
    load_pickle,            # ⬅️ tambahan: untuk load yj_target & mm_target
    resolve_resource_path,  # ⬅️ tambahan: supaya path relatif aman
)
from .prepare_demand_input import prepare_demand_input

CONFIG = load_demand_config()

def run_demand(raw_input: Dict[str, Any]) -> Dict[str, Any]:
    from . import pipeline_production

    # Ambil produksi manual dari frontend
    manual_production = (
        raw_input.get("Produksi (ton)")            # ✅ pakai key yang dikirim app.py
        or raw_input.get("Produksi (ton, dari Model 1)")  # optional fallback
    )

    prod_key = CONFIG.get("dependencies", {}).get("production_output_key", "prediksi_produksi")

    if manual_production not in (None, "", 0):
        # ✅ selalu gunakan produksi yang dikirim backend/frontend
        produksi_prediksi = float(manual_production)
    else:
        # fallback lama, kalau suatu hari dipanggil tanpa produksi
        prod_result = pipeline_production.run_production(raw_input)
        produksi_prediksi = float(prod_result[prod_key])

    # Load model permintaan
    model_path_rel = CONFIG.get("model_path", "demand_model/hybrid_demand.h5")
    
    # Kita panggil fungsi resolve_resource_path (sama seperti di pipeline production)
    # agar path di-resolve penuh mulai dari folder ml_models/penelitian/
    model_path = resolve_resource_path(model_path_rel) 
    
    print(f"[DEBUG] Demand Model Path: {model_path}") # Agar terlihat di terminal jika salah
    model = load_model(model_path)

    # Siapkan input model permintaan
        # Load transformer target (Yeo-Johnson + MinMax) untuk mengembalikan ke satuan ton
    target_cfg = CONFIG.get("target_transform", {})
    yj_target_rel = target_cfg.get("yeo_johnson_path", "demand_model/yj_target.pkl")
    mm_target_rel = target_cfg.get("scaler_path", "demand_model/mm_target.pkl")

    yj_target_path = resolve_resource_path(yj_target_rel)
    mm_target_path = resolve_resource_path(mm_target_rel)

    yj_target = load_pickle(yj_target_path)
    mm_target = load_pickle(mm_target_path)

    x, imputed_features = prepare_demand_input(raw_input, produksi_prediksi)
    y_pred = model.predict(x)

    # Output model masih dalam skala MinMax(YJ(target)) → balikan ke ton
    if isinstance(y_pred, (list, tuple, np.ndarray)):
        y_scaled = float(y_pred[0])
    else:
        y_scaled = float(y_pred)

    # Bentuk array 2D untuk inverse_transform
    y_scaled_arr = np.array([[y_scaled]], dtype="float32")

    # 1) inverse MinMax → balik ke skala Yeo-Johnson
    y_yj_arr = mm_target.inverse_transform(y_scaled_arr)

    # 2) inverse Yeo-Johnson → balik ke skala ton asli
    y_ton_arr = yj_target.inverse_transform(y_yj_arr)

    # Ambil scalar float
    y_val = float(y_ton_arr[0, 0])

    output_field = CONFIG.get("output_schema", {}).get("field_name", "prediksi_permintaan")

    return {
        output_field: y_val,
        prod_key: produksi_prediksi,
        "imputed_features": imputed_features,
    }