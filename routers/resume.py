# C:\Users\ASUS\Documents\All Private Data\Work\Portofolio\Backend\routers\resume.py

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File
from fastapi.responses import Response, FileResponse
from sqlalchemy.orm import Session
from database import SessionLocal
import models, uuid, shutil
import json
import os
from fpdf import FPDF

router = APIRouter(prefix="/api/v1/resume", tags=["Resume Builder"])

UPLOAD_DIR = "uploads"
if not os.path.exists(UPLOAD_DIR):
    os.makedirs(UPLOAD_DIR)
    
def get_db():
    db = SessionLocal()
    try: yield db
    finally: db.close()

class PDFResume(FPDF):
    def section_title(self, title):
        self.set_x(15)
        self.set_font("TNR", "B", 11)
        self.cell(0, 10, title.upper(), ln=True)
        self.set_line_width(0.5)
        self.line(15, self.get_y() - 2, 195, self.get_y() - 2)
        self.set_line_width(0.5)
        self.ln(2)

# ==========================================
# BAGIAN 1: RUTE STATIS (TARUH DI ATAS)
# ==========================================

@router.get("/")
def get_all_resumes(db: Session = Depends(get_db)):
    return db.query(models.ResumeData).order_by(models.ResumeData.updated_at.desc()).all()

@router.post("/upload")
async def upload_resume_pdf(file: UploadFile = File(...), db: Session = Depends(get_db)):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Hanya file PDF yang diperbolehkan")
    
    file_extension = os.path.splitext(file.filename)[1]
    unique_filename = f"{uuid.uuid4()}{file_extension}"
    file_path = os.path.join(UPLOAD_DIR, unique_filename)

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Simpan info ke database agar muncul di daftar admin
    new_resume = models.ResumeData(
        resume_name=file.filename,
        full_name="Uploaded PDF",
        email=file_path # Kita gunakan field email sementara untuk simpan path file
    )
    db.add(new_resume)
    db.commit()
    db.refresh(new_resume)
    return {"message": "File uploaded", "id": new_resume.id}

@router.get("/active/download")
def download_active_resume(db: Session = Depends(get_db)):
    resume = db.query(models.ResumeData).filter(models.ResumeData.is_active == True).first()
    if not resume:
        resume = db.query(models.ResumeData).order_by(models.ResumeData.updated_at.desc()).first()
    
    if not resume:
        raise HTTPException(status_code=404, detail="Data tidak ditemukan")
    
    # Cek apakah ini file upload atau data dinamis
    if resume.email and os.path.exists(resume.email):
        return FileResponse(resume.email, media_type="application/pdf", filename=resume.resume_name)
    
    # Fallback ke generator jika bukan file upload
    from .resume import generate_pdf_logic
    return generate_pdf_logic(resume)

@router.post("/")
def create_resume(data: dict, db: Session = Depends(get_db)):
    new_resume = models.ResumeData(**data)
    db.add(new_resume)
    db.commit()
    db.refresh(new_resume)
    return new_resume

# ==========================================
# BAGIAN 2: RUTE DINAMIS (DENGAN ID)
# ==========================================

@router.get("/{id}")
def get_resume(id: int, db: Session = Depends(get_db)):
    resume = db.query(models.ResumeData).filter(models.ResumeData.id == id).first()
    if not resume: raise HTTPException(status_code=404)
    return resume

@router.get("/generate/{id}")
def preview_pdf_by_id(id: int, db: Session = Depends(get_db)):
    resume = db.query(models.ResumeData).filter(models.ResumeData.id == id).first()
    if not resume: raise HTTPException(status_code=404)
    
    if resume.email and os.path.exists(resume.email):
        return FileResponse(resume.email, media_type="application/pdf")
    
    from .resume import generate_pdf_logic
    return generate_pdf_logic(resume, preview=True)

@router.put("/{id}")
def update_resume(id: int, data: dict, db: Session = Depends(get_db)):
    resume = db.query(models.ResumeData).filter(models.ResumeData.id == id).first()
    if not resume: raise HTTPException(status_code=404)
    for key, value in data.items():
        setattr(resume, key, value)
    db.commit()
    return {"message": "Updated"}

@router.delete("/{id}")
def delete_resume(id: int, db: Session = Depends(get_db)):
    resume = db.query(models.ResumeData).filter(models.ResumeData.id == id).first()
    if resume and resume.email and os.path.exists(resume.email):
        os.remove(resume.email) # Hapus file fisik
    db.query(models.ResumeData).filter(models.ResumeData.id == id).delete()
    db.commit()
    return {"message": "Deleted"}

@router.patch("/{id}/activate")
def activate_resume(id: int, db: Session = Depends(get_db)):
    db.query(models.ResumeData).update({models.ResumeData.is_active: False})
    db.query(models.ResumeData).filter(models.ResumeData.id == id).update({models.ResumeData.is_active: True})
    db.commit()
    return {"message": "Activated"}

# ==========================================
# BAGIAN 3: LOGIKA CORE PDF (KODE LAMA ANDA)
# ==========================================

def generate_pdf_logic(resume, preview: bool = False):
    """Fungsi pembantu yang berisi seluruh logika desain PDF Anda."""
    pdf = PDFResume()
    font_path = os.path.join("assets", "fonts", "TimesNewRoman.ttf")
    pdf.add_font("TNR", "", font_path)
    pdf.add_font("TNR", "B", font_path) 
    
    pdf.add_page()
    pdf.set_margins(15, 15, 15)
    pdf.set_auto_page_break(auto=True, margin=15)
    w_content = 180
    bullet_indent = 20
    bullet_symbol_x = 15

    # Header Nama
    pdf.set_font("TNR", "B", 16)
    pdf.cell(0, 10, resume.full_name.upper(), ln=True, align="C")
    
    pdf.set_font("TNR", "", 9)
    # Kontak
    try:
        # Mengambil data dari field 'phone' yang berisi list of contacts
        contacts = json.loads(resume.phone or "[]")
        
        # Perbaikan: Ambil properti 'url' atau 'value', bukan 'label'
        # Kita filter agar hanya mengambil yang datanya tidak kosong
        contact_values = []
        for c in contacts:
            val = c.get('url') or c.get('value')
            if val:
                contact_values.append(val)
        
        pdf.set_font("TNR", "", 9)
        # Gabungkan dengan separator " | "
        pdf.cell(0, 5, " | ".join(contact_values), ln=True, align="C")
    except Exception as e:
        # Fallback jika JSON bermasalah
        pdf.set_font("TNR", "", 9)
        pdf.cell(0, 5, f"{resume.email}", ln=True, align="C")
    pdf.ln(5)

    # Summary (JUSTIFY FIX)
    pdf.set_x(15)
    pdf.set_font("TNR", "", 10)

    # Bersihkan spasi & newline agar justify sempurna
    clean_summary = " ".join((resume.summary or "").split())
    
    pdf.multi_cell(
        w_content,
        5,
        clean_summary,
        align="J"
    )

    pdf.ln(4)

    # Experience
    pdf.section_title("WORK & RESEARCH EXPERIENCE")
    experiences = json.loads(resume.experience_json or "[]")
    for exp in experiences:
        pdf.set_font("TNR", "B", 10)
        pdf.set_x(15)
        pdf.cell(130, 6, f"{exp.get('company', '')} | {exp.get('role', '')}")
        pdf.set_font("TNR", "", 9)
        pdf.cell(0, 6, exp.get('period', ''), ln=True, align="R")
        
        for bullet in exp.get('bullets', []):
            if bullet.strip():
                pdf.set_x(bullet_symbol_x)
                pdf.write(5, "• ") # Tulis simbol peluru tanpa ganti baris
                pdf.set_x(bullet_indent) # Pindahkan kursor ke posisi indentasi teks
                pdf.multi_cell(w_content - 5, 5, bullet) # Cetak teks dengan lebar yang dikurangi
        pdf.ln(2)

    # Projects
    pdf.section_title("TECHNICAL PROJECTS")
    projects = json.loads(resume.projects_json or "[]")
    for proj in projects:
        pdf.set_font("TNR", "B", 10)
        pdf.set_x(15)
        pdf.cell(0, 6, f"{proj.get('title', '')} | {proj.get('category', '')}", ln=True)
        
        for bullet in proj.get('bullets', []):
            if bullet.strip():
                pdf.set_x(bullet_symbol_x)
                pdf.write(5, "• ")
                pdf.set_x(bullet_indent)
                pdf.multi_cell(w_content - 5, 5, bullet)
        pdf.ln(2)

    # Education
    pdf.section_title("EDUCATIONS")
    education = json.loads(resume.education_json or "[]")
    for edu in education:
        pdf.set_font("TNR", "B", 10)
        pdf.set_x(15)
        pdf.cell(130, 6, edu.get('institution', '').upper())
        pdf.set_font("TNR", "", 9)
        pdf.cell(0, 6, edu.get('period', ''), ln=True, align="R")
        pdf.set_x(15)
        pdf.set_font("TNR", "", 10)
        pdf.cell(0, 5, f"{edu.get('degree', '')}", ln=True)
        
        for bullet in edu.get('bullets', []):
            if bullet.strip():
                pdf.set_x(bullet_symbol_x)
                pdf.write(5, "• ")
                pdf.set_x(bullet_indent)
                pdf.multi_cell(w_content - 5, 5, bullet)
        pdf.ln(2)

    # Skills
    pdf.section_title("SKILLS")
    try:
        skill_groups = json.loads(resume.skills_json or "[]")
        for group in skill_groups:
            pdf.set_font("TNR", "B", 9)
            pdf.write(5, f"{group.get('category', '')}: ")
            pdf.set_font("TNR", "", 9)
            pdf.write(5, f"{group.get('summary', '')}\n")
            pdf.ln(2)
    except: pass

    # Certifications
    pdf.section_title("TRAININGS AND CERTIFICATIONS")
    try:
        certs = json.loads(resume.github_url or "[]")
        for cert in certs:
            if cert.strip():
                pdf.set_x(bullet_symbol_x)
                pdf.write(5, "• ")
                pdf.set_x(bullet_indent)
                pdf.multi_cell(w_content - 5, 5, cert)
    except: pass

    pdf_content = pdf.output(dest='S')
    disposition = "inline" if preview else "attachment"
    safe_name = (resume.full_name or "Resume").replace(" ", "_")
    filename = f"CV_{safe_name}.pdf"

    return Response(
        content=bytes(pdf_content), 
        media_type="application/pdf",
        headers={
            "Content-Disposition": f"{disposition}; filename={filename}",
            "Cache-Control": "no-cache"
        }
    )