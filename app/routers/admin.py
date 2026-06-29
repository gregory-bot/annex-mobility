"""Admin + simulation endpoints."""
from __future__ import annotations

from fastapi import APIRouter, Depends
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.bot.engine import handle_message
from app.db import get_session
from app.models.entities import Driver, Trip, User
from app.services.ai_pricing import get_all_quotes

router = APIRouter(prefix="/admin", tags=["admin"])


class SimulateIn(BaseModel):
    phone: str
    body: str
    channel: str = "whatsapp"
    latitude: float | None = None
    longitude: float | None = None


class QuoteRequest(BaseModel):
    pickup_address: str
    dropoff_address: str
    pickup_lat: float
    pickup_lng: float
    drop_lat: float
    drop_lng: float


@router.post("/simulate")
async def simulate(payload: SimulateIn, db: AsyncSession = Depends(get_session)):
    """Send a message as a user — returns bot reply. No Twilio needed."""
    reply = await handle_message(
        db,
        phone=payload.phone,
        channel=payload.channel,
        body=payload.body,
        latitude=payload.latitude,
        longitude=payload.longitude,
    )
    return {"reply": reply.text}


@router.post("/quotes")
async def get_quotes(req: QuoteRequest):
    """Directly fetch AI price quotes for a route (for testing)."""
    quotes = await get_all_quotes(
        pickup_address=req.pickup_address,
        dropoff_address=req.dropoff_address,
        pickup_lat=req.pickup_lat,
        pickup_lng=req.pickup_lng,
        drop_lat=req.drop_lat,
        drop_lng=req.drop_lng,
    )
    return [
        {
            "platform": q.platform,
            "name": q.name,
            "type": q.type,
            "price_kes": q.price_kes,
            "duration_min": q.duration_min,
            "distance_km": q.distance_km,
            "surge": q.surge,
            "source": q.source,
            "deep_link_web": q.deep_link_web,
            "is_cheapest": q.is_cheapest,
        }
        for q in quotes
    ]


@router.get("/users")
async def list_users(db: AsyncSession = Depends(get_session)):
    rows = (await db.execute(select(User).order_by(User.id.desc()).limit(100))).scalars().all()
    return [{"id": u.id, "phone": u.phone, "channel": u.channel, "name": u.name} for u in rows]


@router.get("/trips")
async def list_trips(db: AsyncSession = Depends(get_session)):
    rows = (await db.execute(select(Trip).order_by(Trip.id.desc()).limit(100))).scalars().all()
    return [
        {
            "id": t.id,
            "user_id": t.user_id,
            "driver_id": t.driver_id,
            "status": t.status,
            "platform": t.chosen_platform_name,
            "pickup": t.pickup_text,
            "dropoff": t.dropoff_text,
            "km": t.distance_km,
            "min": t.duration_min,
            "fare_kes": t.fare_kes,
        }
        for t in rows
    ]


@router.get("/drivers")
async def list_drivers(db: AsyncSession = Depends(get_session)):
    rows = (await db.execute(select(Driver))).scalars().all()
    return [
        {
            "id": d.id,
            "name": d.name,
            "plate": d.plate,
            "vehicle": d.vehicle,
            "type": d.vehicle_type,
            "rating": d.rating,
            "available": d.available,
            "phone": d.phone,
        }
        for d in rows
    ]


@router.post("/drivers/{driver_id}/release")
async def release_driver(driver_id: int, db: AsyncSession = Depends(get_session)):
    """Mark a driver available again (for testing)."""
    from app.services import drivers as drv_svc
    await drv_svc.release(db, driver_id)
    return {"ok": True, "driver_id": driver_id}
