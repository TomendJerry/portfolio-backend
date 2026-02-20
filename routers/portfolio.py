from fastapi import APIRouter, HTTPException, Depends, status
from sqlalchemy.orm import Session
from typing import List, Optional, Any
from pydantic import BaseModel
import datetime
import json

# Import koneksi database dan model
from database import SessionLocal
import models

router = APIRouter(
    prefix="/api/v1/portfolio",
    tags=["Portfolio Management"]
)

# --- Dependency untuk mendapatkan Session Database ---
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# --- Schema Data (Pydantic) ---

class ProjectBase(BaseModel):
    title: str
    category: str
    description: Optional[str] = None
    tech_stack: Optional[str] = None
    # TAMBAHKAN 3 BARIS INI
    problem_statement: Optional[str] = None
    objective: Optional[str] = None
    approach: Optional[str] = None
    metrics_json: Optional[Any] = None
    flowchart_json: Optional[Any] = None
    
    documentation_url: Optional[str] = None
    has_live_demo: bool = False
    thumbnail_url: Optional[str] = None

class ProjectCreate(ProjectBase):
    pass

class ProjectRead(ProjectBase):
    id: str
    created_at: datetime.datetime
    metrics_json: Optional[Any] = None
    class Config:
        from_attributes = True

# --- Endpoint API ---

@router.get("/projects", response_model=List[ProjectRead])
def get_all_projects(db: Session = Depends(get_db)):
    """
    Mengambil semua daftar proyek untuk ditampilkan di halaman utama portfolio.
    """
    projects = db.query(models.Project).order_by(models.Project.created_at.desc()).all()
    return projects

@router.get("/projects/{project_id}", response_model=ProjectRead)
def get_project_detail(project_id: str, db: Session = Depends(get_db)):
    """
    Mengambil detail satu proyek spesifik berdasarkan ID.
    """
    project = db.query(models.Project).filter(models.Project.id == project_id).first()
    if not project:
        raise HTTPException(status_code=404, detail="Proyek tidak ditemukan")
    return project

@router.post("/projects", response_model=ProjectRead, status_code=status.HTTP_201_CREATED)
def add_new_project(project: ProjectCreate, db: Session = Depends(get_db)):
    project_data = project.dict()
    
    # Logic untuk Metrics JSON
    if project_data.get("metrics_json") and not isinstance(project_data["metrics_json"], str):
        project_data["metrics_json"] = json.dumps(project_data["metrics_json"])
    
    # TAMBAHKAN LOGIC INI: Untuk Flowchart JSON
    if project_data.get("flowchart_json") and not isinstance(project_data["flowchart_json"], str):
        project_data["flowchart_json"] = json.dumps(project_data["flowchart_json"])
        
    new_project = models.Project(**project_data)
    db.add(new_project)
    db.commit()
    db.refresh(new_project)
    return new_project

@router.delete("/projects/{project_id}")
def delete_project(project_id: str, db: Session = Depends(get_db)):
    """
    Menghapus dokumentasi proyek dari database.
    """
    project = db.query(models.Project).filter(models.Project.id == project_id).first()
    if not project:
        raise HTTPException(status_code=404, detail="Proyek tidak ditemukan")
    
    db.delete(project)
    db.commit()
    return {"message": f"Proyek '{project.title}' berhasil dihapus"}