from __future__ import annotations

from datetime import datetime
from typing import Optional

from sqlalchemy import DateTime, Float, ForeignKey, Integer, String, Text, func, Index
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db import Base


class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    phone: Mapped[str] = mapped_column(String(32), unique=True, index=True)
    channel: Mapped[str] = mapped_column(String(16), default="whatsapp")
    name: Mapped[Optional[str]] = mapped_column(String(120), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())

    sessions: Mapped[list["Session"]] = relationship(back_populates="user")
    trips: Mapped[list["Trip"]] = relationship(back_populates="user")


class Session(Base):
    __tablename__ = "sessions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), index=True)
    state: Mapped[str] = mapped_column(String(40), default="idle")
    data: Mapped[str] = mapped_column(Text, default="{}")
    updated_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now(), onupdate=func.now())

    user: Mapped[User] = relationship(back_populates="sessions")


class Driver(Base):
    __tablename__ = "drivers"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(120))
    phone: Mapped[str] = mapped_column(String(32))
    plate: Mapped[str] = mapped_column(String(16))
    vehicle: Mapped[str] = mapped_column(String(120))
    rating: Mapped[float] = mapped_column(Float, default=4.8)
    available: Mapped[bool] = mapped_column(default=True)
    vehicle_type: Mapped[str] = mapped_column(String(20), default="car")


class Trip(Base):
    __tablename__ = "trips"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), index=True)
    driver_id: Mapped[Optional[int]] = mapped_column(ForeignKey("drivers.id"), nullable=True)

    pickup_text: Mapped[str] = mapped_column(String(255))
    pickup_lat: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    pickup_lng: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    dropoff_text: Mapped[str] = mapped_column(String(255))
    dropoff_lat: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    dropoff_lng: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    distance_km: Mapped[float] = mapped_column(Float, default=0)
    duration_min: Mapped[float] = mapped_column(Float, default=0)
    fare_kes: Mapped[float] = mapped_column(Float, default=0)

    chosen_platform: Mapped[Optional[str]] = mapped_column(String(32), nullable=True)
    chosen_platform_name: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)

    status: Mapped[str] = mapped_column(String(32), default="requested")
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    completed_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)

    user: Mapped[User] = relationship(back_populates="trips")
