"""Driver matching service.

Seeded with a realistic mock fleet covering car and motorbike types.
Replace find_available() with a real partner API call in production.
"""
from __future__ import annotations

import random
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.entities import Driver

_SEED_DRIVERS = [
    # Cars
    {"name": "James Mwangi", "phone": "+254711000001", "plate": "KBZ 123A", "vehicle": "Toyota Axio White", "rating": 4.9, "vehicle_type": "car"},
    {"name": "Grace Njeri",  "phone": "+254711000002", "plate": "KCB 456B", "vehicle": "Nissan Tiida Silver","rating": 4.8, "vehicle_type": "car"},
    {"name": "Peter Omondi", "phone": "+254711000003", "plate": "KDA 789C", "vehicle": "Toyota Fielder Grey", "rating": 4.7, "vehicle_type": "car"},
    {"name": "Faith Wangari", "phone": "+254711000004", "plate": "KDB 321D", "vehicle": "Honda Fit Blue",    "rating": 4.9, "vehicle_type": "car"},
    {"name": "Brian Kamau",  "phone": "+254711000005", "plate": "KCA 654E", "vehicle": "Toyota Vitz White",  "rating": 4.6, "vehicle_type": "car"},
    {"name": "Alice Auma",   "phone": "+254711000006", "plate": "KDD 987F", "vehicle": "Mazda Demio Red",    "rating": 4.8, "vehicle_type": "car"},
    {"name": "Samuel Otieno","phone": "+254711000007", "plate": "KCF 111G", "vehicle": "Toyota Belta Black", "rating": 4.7, "vehicle_type": "car"},
    {"name": "Lucy Nduta",   "phone": "+254711000008", "plate": "KBF 222H", "vehicle": "Subaru Impreza Maroon","rating": 4.9, "vehicle_type": "car"},
    # Motorbikes (Boda)
    {"name": "Kevin Maina",  "phone": "+254722000001", "plate": "KMCA 01B", "vehicle": "Bajaj Boxer Black",  "rating": 4.7, "vehicle_type": "motorbike"},
    {"name": "David Waweru", "phone": "+254722000002", "plate": "KMCB 02C", "vehicle": "TVS Apache Blue",    "rating": 4.6, "vehicle_type": "motorbike"},
    {"name": "Mercy Chebet", "phone": "+254722000003", "plate": "KMCC 03D", "vehicle": "Yamaha Crux Red",    "rating": 4.8, "vehicle_type": "motorbike"},
    {"name": "Tom Kipchoge", "phone": "+254722000004", "plate": "KMCD 04E", "vehicle": "Honda CG125 Black",  "rating": 4.5, "vehicle_type": "motorbike"},
]


async def seed(db: AsyncSession) -> None:
    existing = (await db.execute(select(Driver).limit(1))).scalar_one_or_none()
    if existing:
        return
    for d in _SEED_DRIVERS:
        db.add(Driver(**d))
    await db.commit()


async def find_available(db: AsyncSession, vehicle_type: str = "car") -> Driver | None:
    """Find an available driver. Prefer the specified vehicle_type but fall back."""
    rows = (
        await db.execute(
            select(Driver)
            .where(Driver.available == True, Driver.vehicle_type == vehicle_type)
        )
    ).scalars().all()
    if not rows:
        # Fallback: any available driver
        rows = (
            await db.execute(select(Driver).where(Driver.available == True))
        ).scalars().all()
    if not rows:
        return None
    drv = random.choice(rows)
    drv.available = False
    await db.commit()
    return drv


async def release(db: AsyncSession, driver_id: int) -> None:
    drv = await db.get(Driver, driver_id)
    if drv:
        drv.available = True
        await db.commit()
