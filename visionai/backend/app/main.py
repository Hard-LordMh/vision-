import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

import app.models.schema
from app.database.database import Base, engine

# Create all tables
Base.metadata.create_all(bind=engine)

app = FastAPI(title="VisionAI API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files directory for sample telemetry and datasets
static_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "storage", "static")
os.makedirs(static_dir, exist_ok=True)
app.mount("/static", StaticFiles(directory=static_dir), name="static")

@app.get("/api/health")
def health_check():
    return {"status": "ok"}

from app.api.endpoints import router

app.include_router(router, prefix="/api")

