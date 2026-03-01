# Backend/routers/rating.py
from fastapi import APIRouter, Depends, Request, HTTPException
from sqlalchemy.orm import Session
from database import SessionLocal
import models
import datetime

router = APIRouter(prefix="/api/v1/rating", tags=["User Feedback"])

def get_db():
    db = SessionLocal()
    try: yield db
    finally: db.close()

# --- ENDPOINT PUBLIK: KIRIM RATING ---
@router.post("/")
async def submit_rating(request: Request, data: dict, db: Session = Depends(get_db)):
    client_ip = request.client.host 
    # Cek duplikasi IP dalam 24 jam
    day_ago = datetime.datetime.utcnow() - datetime.timedelta(days=1)
    already_rated = db.query(models.Rating).filter(
        models.Rating.user_ip == client_ip,
        models.Rating.created_at >= day_ago
    ).first()
    
    if already_rated:
        raise HTTPException(status_code=429, detail="Satu IP hanya boleh memberi rating 1x sehari")

    new_rating = models.Rating(
        score=data['score'],
        comment=data.get('comment'),
        category=data.get('category', 'General'),
        user_ip=client_ip
    )
    db.add(new_rating)
    db.commit()
    return {"message": "Rating submitted"}

# --- ENDPOINT ADMIN: AMBIL SEMUA DATA (Hanya Admin/Super Admin) ---
@router.get("/admin/all")
async def get_all_ratings(role: str, db: Session = Depends(get_db)):
    # Normalisasi role menjadi huruf kecil dan ganti spasi dengan underscore
    normalized_role = role.lower().replace(" ", "_")
    
    # Izinkan akses jika role adalah admin atau super_admin
    if normalized_role not in ["admin", "super_admin"]:
        raise HTTPException(status_code=403, detail="Akses ditolak")
    
    return db.query(models.Rating).order_by(models.Rating.created_at.desc()).all()

@router.delete("/admin/{rating_id}")
async def delete_rating(rating_id: int, role: str, db: Session = Depends(get_db)):
    normalized_role = role.lower().replace(" ", "_")
    
    if normalized_role != "super_admin":
        raise HTTPException(status_code=403, detail="Hanya Super Admin yang bisa menghapus")
    
    db.query(models.Rating).filter(models.Rating.id == rating_id).delete()
    db.commit()
    return {"message": "Rating deleted"}