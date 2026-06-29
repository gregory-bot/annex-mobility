"""Annex Mobility — WhatsApp & SMS ride aggregator backend."""
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.db import init_db
from app.routers import admin, africastalking, whatsapp
from app.services import drivers as driver_svc
from app.db import async_session_factory


@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_db()
    # Seed mock drivers on startup
    async with async_session_factory() as db:
        await driver_svc.seed(db)
    yield


app = FastAPI(
    title="Annex Mobility",
    version="2.0.0",
    description=(
        "Annex Mobility lets people book, manage, and track rides from multiple transport providers through WhatsApp or SMS "
    ),
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", tags=["meta"])
def root():
    return {
        "name": "Annex Mobility",
        "version": "2.0.0",
        "status": "ok",
        "docs": "/docs",
        "features": [
            "Gemini AI price comparison (Uber, Bolt, Little, Faras, Yego)",
            "WhatsApp + SMS booking",
            "Deep-link to platform apps",
            "Real-time geocoding",
        ],
    }


@app.get("/healthz", tags=["meta"])
def healthz():
    return {"ok": True}


app.include_router(whatsapp.router)
app.include_router(africastalking.router)
app.include_router(admin.router)
