from fastapi import APIRouter, Depends, Request, HTTPException
from sqlalchemy.orm import Session
from datetime import datetime, timezone
import uuid
import models  # Pastikan file models.py Anda sudah ada di folder yang sama
from database import get_db # Import fungsi get_db Anda

router = APIRouter(
    prefix="/audit",
    tags=["Security Audit"]
)

@router.post("/log-access")
async def log_visitor_access(
    request: Request, 
    data: dict, 
    db: Session = Depends(get_db)
):
    try:
        # 1. Ambil data otomatis dari Server-Side (Koneksi)
        client_ip = request.client.host 
        user_agent = request.headers.get("user-agent", "Unknown")
        
        # 2. Ambil data dari Payload Frontend (Fingerprint)
        # Data ini dikirim oleh library FingerprintJS di sisi Client
        visitor_id = data.get("visitorId")
        browser_details = data.get("browser_details", {})
        action_taken = data.get("action", "PAGE_VISIT")

        if not visitor_id:
            raise HTTPException(status_code=400, detail="Visitor ID is required for audit")

        # 3. Buat Record Audit Baru
        new_audit_log = models.SecurityAudit(
            id=str(uuid.uuid4()),
            timestamp=datetime.now(timezone.utc), 
            ip_address=client_ip,
            device_id=visitor_id,
            user_agent=user_agent,
            browser_data=browser_details, # Disimpan sebagai JSON di DB
            action=action_taken,
            status_code="200"
        )

        # 4. Simpan ke Database Neon
        db.add(new_audit_log)
        db.commit()
        db.refresh(new_audit_log)

        return {
            "status": "success", 
            "message": "Access logged for security audit",
            "log_id": new_audit_log.id
        }

    except Exception as e:
        db.rollback()
        print(f"Audit Error: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to log security audit")

@router.get("/logs")
async def get_audit_logs(db: Session = Depends(get_db)):
    # Endpoint ini nanti akan dipanggil oleh Menu Sidebar Admin Anda
    logs = db.query(models.SecurityAudit).order_by(models.SecurityAudit.timestamp.desc()).all()
    return logs