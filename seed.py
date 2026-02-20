import pandas as pd
import os
from sqlalchemy.orm import Session
from database import SessionLocal, engine
import models

# Mapping Path File (Sesuai lokasi di komputer Anda)
BASE_PATH = r"C:\Users\ASUS\Documents\All Private Data\Work\Portofolio\Backend\ml_models\skripsi\Dataset"

FILES = {
    "rice_production": os.path.join(BASE_PATH, "production_data.csv"),
    "rice_demand": os.path.join(BASE_PATH, "demand_data.csv"),
    "rice_divre": os.path.join(BASE_PATH, "divre_predictions.csv"),
    "mng_pemetaan": [
        os.path.join(BASE_PATH, "Pemetaan", "data_final_untuk_model.csv"),
        os.path.join(BASE_PATH, "Pemetaan", "klasifikasi_titik_dengan_model.csv")
    ],
    "mng_prediksi": [
        os.path.join(BASE_PATH, "Prediksi", "klasifikasi_dengan_spesies_dbh.csv"),
        os.path.join(BASE_PATH, "Prediksi", "output_prediksi_agb_agc.csv")
    ]
}

def clean_decimal(val):
    """Mengubah format koma (0,06) menjadi titik (0.06) untuk database."""
    if isinstance(val, str):
        return float(val.replace(',', '.'))
    return val

def clear_data(db: Session):
    """Menghapus data lama sebelum migrasi baru dimulai agar tidak duplikat."""
    print("Membersihkan data lama di database...")
    db.query(models.RiceProduction).delete()
    db.query(models.RiceDemand).delete()
    db.query(models.RiceDivrePrediction).delete()
    db.query(models.MangrovePemetaan).delete()
    db.query(models.MangrovePrediksi).delete()
    db.commit()

def seed_rice(db: Session):
    print("Migrasi Data Rice (Production, Demand, & Divre)...")
    
    # 1. Production Data
    df_prod = pd.read_csv(FILES["rice_production"])
    for _, row in df_prod.iterrows():
        db_row = models.RiceProduction(
            provinsi=row['Provinsi'], tahun=row['Tahun'],
            luas_panen_ha=row['Luas Panen (ha)'], produksi_ton=row['Produksi (ton)'],
            suhu_min=row['Suhu Min (oC)'], suhu_rata=row['Suhu Rata-rata (oC)'],
            suhu_maks=row['Suhu Maks (oC)'], kelembapan_persen=row['Kelembapan (%)'],
            curah_hujan_mm=row['Curah Hujan (mm)'], hari_hujan=row['Jumlah Hari Hujan (hari)'],
            kecepatan_angin=row['Kecepatan Angin (m/s)'], tekanan_udara=row['Tekanan Udara (mb)'],
            penyinaran_matahari=row['Penyinaran Matahari (%)'], annual_soi_bom=row['Annual_SOI_BOM'],
            soi_bom_label=row['SOI_BOM_Label'], annual_dmi=row['Annual_DMI'],
            produktivitas_ton_ha=row['Produktivitas (ton/ha)'], data_label=row['Data Label']
        )
        db.add(db_row)

    # 2. Demand Data
    df_dem = pd.read_csv(FILES["rice_demand"])
    for _, row in df_dem.iterrows():
        db_row = models.RiceDemand(
            provinsi=row['Provinsi'], tahun=row['Tahun'],
            total_konsumsi_ton=row['Total Konsumsi Beras (Ton)'],
            jumlah_penduduk=row['Jumlah Penduduk (Jiwa)'], laju_pertumbuhan=row['Laju Pertumbuhan Penduduk (%)'],
            harga_beras=row['Harga Beras (Rp/Kg)'], harga_jagung=row['Harga Jagung (Rp/Kg)'],
            harga_mie_instan=row['Harga Mie Instan (Rp/Bungkus)'], tingkat_kemiskinan=row['Tingkat Kemiskinan (%)'],
            garis_kemiskinan=row['Garis Kemiskinan per Kapita (Rp/tahun)'],
            pdrb_perkapita=row['Produk Domestik Regional Bruto per Kapita HK (Ribu Rp)'],
            gini_rasio=row['Gini Rasio'], rata_anggota_rt=row['Rata-rata Banyaknya Anggota Rumah Tangga (Jiwa)'],
            tpt_persen=row['Tingkat Pengangguran Terbuka (%)'], data_label=row['Data Label']
        )
        db.add(db_row)

    # 3. Divre Predictions (DATA INI YANG SEBELUMNYA TERLEWAT)
    df_divre = pd.read_csv(FILES["rice_divre"])
    for _, row in df_divre.iterrows():
        db_row = models.RiceDivrePrediction(
            divre_name=row['divre_name'],
            base_year=row['base_year'],
            horizon_years=row['horizon_years'],
            target_year=row['target_year'],
            predicted_production=row['predicted_production'],
            predicted_demand=row['predicted_demand']
        )
        db.add(db_row)
        
    db.commit()

def seed_mangrove(db: Session):
    print("Migrasi Data Mangrove (Pemetaan & Prediksi)...")
    
    # 1. Pemetaan (Gabungan 2 File)
    for path in FILES["mng_pemetaan"]:
        df = pd.read_csv(path)
        for _, row in df.iterrows():
            db_row = models.MangrovePemetaan(
                latitude=row['Latitude'], longitude=row['Longitude'],
                b1=row['B1'], b2=row['B2'], b3=row['B3'], b4=row['B4'],
                b5=row['B5'], b6=row['B6'], b7=row['B7'], b8=row['B8'],
                b8a=row['B8A'], b9=row['B9'], b11=row['B11'], b12=row['B12'],
                ndvi=row['NDVI'], evi=row['EVI'], savi=row['SAVI'],
                ground_truth_class=row['Class'],
                predicted_class=row.get('Predicted_Class')
            )
            db.add(db_row)

    # 2. Prediksi (Spesial handling untuk titik koma dan desimal koma)
    df_output = pd.read_csv(FILES["mng_prediksi"][1], sep=';')
    for _, row in df_output.iterrows():
        db_row = models.MangrovePrediksi(
            latitude=clean_decimal(row['Latitude']), longitude=clean_decimal(row['Longitude']),
            acquisition_date=row['Acquisition_Date'],
            predicted_class=row['Predicted_Class'], dominant_species=row['Dominant_Species'],
            species_code=row['Species_Code'],
            predicted_dbh_cm=clean_decimal(row['Predicted_DBH_cm']),
            predicted_agb_kg=clean_decimal(row['Predicted_AGB_kg']),
            predicted_agc_kg=clean_decimal(row['Predicted_AGC_kg']),
            ndvi=clean_decimal(row['NDVI']), evi=clean_decimal(row['EVI']), savi=clean_decimal(row['SAVI'])
        )
        db.add(db_row)
    db.commit()

if __name__ == "__main__":
    db = SessionLocal()
    try:
        # Sinkronisasi tabel
        models.Base.metadata.create_all(bind=engine)
        
        # 1. Hapus data lama
        clear_data(db)
        
        # 2. Isi data baru
        seed_rice(db)
        seed_mangrove(db)
        
        print("-" * 30)
        print("MIGRASI SELESAI DENGAN SUKSES!")
        print("-" * 30)
    except Exception as e:
        print(f"Error saat migrasi: {e}")
        db.rollback()
    finally:
        db.close()