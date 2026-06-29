"""Gemini AI-powered ride price estimation for multiple platforms.

This module uses Google Gemini 2.5 Flash to estimate realistic KES prices
across Uber, Bolt, Little, Yego, Faras, and Bolt Bike — based on distance,
time of day, and Kenyan market pricing (2025).

When GEMINI_API_KEY is not set, it falls back to the deterministic formula.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import math
import time
from dataclasses import dataclass, field
from typing import Optional

import httpx

from app.config import settings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Simple in-process TTL cache (15 minutes) — avoids hammering Gemini on
# repeated identical routes. Replace with Redis at scale.
# ---------------------------------------------------------------------------
_cache: dict[str, tuple[float, list]] = {}  # key -> (timestamp, result)
_CACHE_TTL = 900  # seconds


def _cache_key(pickup_lat: float, pickup_lng: float, drop_lat: float, drop_lng: float) -> str:
    # Round to ~500m precision so nearby requests reuse cache
    raw = f"{round(pickup_lat, 2)},{round(pickup_lng, 2)}-{round(drop_lat, 2)},{round(drop_lng, 2)}"
    return hashlib.md5(raw.encode()).hexdigest()


def _get_cache(key: str) -> Optional[list]:
    entry = _cache.get(key)
    if entry and (time.time() - entry[0]) < _CACHE_TTL:
        return entry[1]
    _cache.pop(key, None)
    return None


def _set_cache(key: str, data: list) -> None:
    _cache[key] = (time.time(), data)
    # Evict old entries to prevent unbounded memory growth
    if len(_cache) > 5000:
        oldest = sorted(_cache.items(), key=lambda x: x[1][0])[:1000]
        for k, _ in oldest:
            _cache.pop(k, None)


# ---------------------------------------------------------------------------
# Haversine distance
# ---------------------------------------------------------------------------
def _haversine(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
    R = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = math.radians(lat2 - lat1), math.radians(lng2 - lng1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


def _duration_min(distance_km: float) -> float:
    """Estimate travel time based on time of day."""
    hour = time.localtime().tm_hour  # server local time (set TZ=Africa/Nairobi in env)
    is_peak = (7 <= hour <= 10) or (17 <= hour <= 20)
    avg_speed = 15.0 if is_peak else 25.0
    return max((distance_km / avg_speed) * 60.0, 3.0)


# ---------------------------------------------------------------------------
# Deterministic formula fallback (no AI needed)
# ---------------------------------------------------------------------------
_PLATFORM_MODELS = {
    "uber":       {"base": 100, "per_km": 50,  "per_min": 3.0,  "min_fare": 150, "peak_surge": 1.25},
    "bolt":       {"base": 80,  "per_km": 45,  "per_min": 2.5,  "min_fare": 130, "peak_surge": 1.15},
    "little":     {"base": 90,  "per_km": 48,  "per_min": 2.7,  "min_fare": 135, "peak_surge": 1.20},
    "faras":      {"base": 85,  "per_km": 47,  "per_min": 2.8,  "min_fare": 140, "peak_surge": 1.18},
    "yego":       {"base": 50,  "per_km": 35,  "per_min": 2.0,  "min_fare": 100, "peak_surge": 1.10},
    "bolt_bike":  {"base": 40,  "per_km": 32,  "per_min": 1.8,  "min_fare": 90,  "peak_surge": 1.10},
}

_PLATFORM_LABELS = {
    "uber": "Uber",
    "bolt": "Bolt",
    "little": "Little",
    "faras": "Faras",
    "yego": "Yego (Boda)",
    "bolt_bike": "Bolt Boda",
}

_PLATFORM_TYPES = {
    "uber": "car",
    "bolt": "car",
    "little": "car",
    "faras": "car",
    "yego": "motorbike",
    "bolt_bike": "motorbike",
}


def _formula_prices(distance_km: float, duration_min: float) -> list[dict]:
    hour = time.localtime().tm_hour
    is_peak = (7 <= hour <= 10) or (17 <= hour <= 20)
    results = []
    for platform, m in _PLATFORM_MODELS.items():
        surge = m["peak_surge"] if is_peak else 1.0
        price = max(m["base"] + distance_km * m["per_km"] + duration_min * m["per_min"], m["min_fare"])
        price = round(price * surge / 10) * 10  # round to nearest 10 KES
        results.append({
            "platform": platform,
            "name": _PLATFORM_LABELS[platform],
            "type": _PLATFORM_TYPES[platform],
            "price_kes": int(price),
            "duration_min": round(duration_min),
            "distance_km": round(distance_km, 2),
            "surge": is_peak,
            "source": "formula",
        })
    return sorted(results, key=lambda x: x["price_kes"])


# ---------------------------------------------------------------------------
# Gemini AI estimation
# ---------------------------------------------------------------------------
async def _call_gemini(
    pickup_address: str,
    dropoff_address: str,
    pickup_lat: float,
    pickup_lng: float,
    drop_lat: float,
    drop_lng: float,
    distance_km: float,
    duration_min: float,
) -> Optional[list[dict]]:
    """Call Gemini 2.5 Flash to get price estimates. Returns None on failure."""
    if not settings.GEMINI_API_KEY:
        return None

    hour = time.localtime().tm_hour
    is_peak = (7 <= hour <= 10) or (17 <= hour <= 20)
    time_str = time.strftime("%H:%M")

    prompt = f"""You are a Kenyan ride-hailing price expert with up-to-date knowledge of 2025 market rates.

Calculate realistic prices in KES for a ride from:
FROM: {pickup_address} (lat={pickup_lat:.5f}, lng={pickup_lng:.5f})
TO:   {dropoff_address} (lat={drop_lat:.5f}, lng={drop_lng:.5f})

Computed distance: {distance_km:.2f} km
Estimated travel time: {duration_min:.0f} minutes
Current time: {time_str} {"(PEAK HOUR)" if is_peak else "(off-peak)"}

Pricing rules for Kenya 2025:
- Uber: Base KES 100 + KES 50/km + KES 3/min. Peak surge +25%. Min KES 150.
- Bolt: Base KES 80 + KES 45/km + KES 2.5/min. Peak surge +15%. Min KES 130.
- Little: Base KES 90 + KES 48/km + KES 2.7/min. Peak surge +20%. Min KES 135.
- Faras: Base KES 85 + KES 47/km + KES 2.8/min. Peak surge +18%. Min KES 140.
- Yego (motorbike): Base KES 50 + KES 35/km + KES 2/min. Peak surge +10%. Min KES 100.
- Bolt Boda (motorbike): Base KES 40 + KES 32/km + KES 1.8/min. Peak surge +10%. Min KES 90.

Add slight realistic variance (±5-15 KES) between similar platforms.
For trips over 10km, cars become more economical vs motorbikes.
For trips under 5km, motorbikes may be faster and cheaper.

Return ONLY valid JSON, no markdown, no explanation:
{{
  "uber": {{"price": <int>, "available": true}},
  "bolt": {{"price": <int>, "available": true}},
  "little": {{"price": <int>, "available": true}},
  "faras": {{"price": <int>, "available": true}},
  "yego": {{"price": <int>, "available": true}},
  "bolt_bike": {{"price": <int>, "available": true}}
}}"""

    url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent"
    headers = {"Content-Type": "application/json"}
    params = {"key": settings.GEMINI_API_KEY}
    body = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0.2, "maxOutputTokens": 512},
    }

    try:
        async with httpx.AsyncClient(timeout=15) as client:
            r = await client.post(url, headers=headers, params=params, json=body)
            r.raise_for_status()
            data = r.json()

        text = data["candidates"][0]["content"]["parts"][0]["text"]
        # Strip markdown fences if present
        text = text.strip().lstrip("```json").lstrip("```").rstrip("```").strip()
        parsed = json.loads(text)

        results = []
        for platform, info in parsed.items():
            if not info.get("available", True):
                continue
            results.append({
                "platform": platform,
                "name": _PLATFORM_LABELS.get(platform, platform.title()),
                "type": _PLATFORM_TYPES.get(platform, "car"),
                "price_kes": int(info["price"]),
                "duration_min": round(duration_min),
                "distance_km": round(distance_km, 2),
                "surge": is_peak,
                "source": "ai",
            })
        return sorted(results, key=lambda x: x["price_kes"])

    except Exception as exc:
        logger.warning("Gemini pricing failed (%s), using formula fallback", exc)
        return None


# ---------------------------------------------------------------------------
# Deep-link URLs for each platform
# ---------------------------------------------------------------------------
def get_deep_links(platform: str, pickup_lat: float, pickup_lng: float,
                   drop_lat: float, drop_lng: float) -> dict[str, str]:
    p = {"lat": pickup_lat, "lng": pickup_lng}
    d = {"lat": drop_lat, "lng": drop_lng}
    links: dict[str, dict[str, str]] = {
        "uber": {
            "app": (
                f"uber://?action=setPickup"
                f"&pickup[latitude]={p['lat']}&pickup[longitude]={p['lng']}"
                f"&dropoff[latitude]={d['lat']}&dropoff[longitude]={d['lng']}"
            ),
            "web": (
                f"https://m.uber.com/ul/?action=setPickup"
                f"&pickup[latitude]={p['lat']}&pickup[longitude]={p['lng']}"
                f"&dropoff[latitude]={d['lat']}&dropoff[longitude]={d['lng']}"
            ),
        },
        "bolt": {
            "app": f"bolt://ridepicker?pickup={p['lat']},{p['lng']}&destination={d['lat']},{d['lng']}",
            "web": (
                f"https://bolt.eu/en/ride/?pickup_latitude={p['lat']}&pickup_longitude={p['lng']}"
                f"&destination_latitude={d['lat']}&destination_longitude={d['lng']}"
            ),
        },
        "little": {
            "app": f"little://ride?pickup={p['lat']},{p['lng']}&destination={d['lat']},{d['lng']}",
            "web": (
                f"https://little.africa/ride?pickup_lat={p['lat']}&pickup_lng={p['lng']}"
                f"&dropoff_lat={d['lat']}&dropoff_lng={d['lng']}"
            ),
        },
        "faras": {
            "app": f"faras://book?pickup={p['lat']},{p['lng']}&dest={d['lat']},{d['lng']}",
            "web": (
                f"https://faras.co.ke/book?pickup_lat={p['lat']}&pickup_lng={p['lng']}"
                f"&dest_lat={d['lat']}&dest_lng={d['lng']}"
            ),
        },
        "yego": {
            "app": f"yego://ride?pickup={p['lat']},{p['lng']}&destination={d['lat']},{d['lng']}",
            "web": (
                f"https://yego.co.ke/ride?pickup_lat={p['lat']}&pickup_lng={p['lng']}"
                f"&dropoff_lat={d['lat']}&dropoff_lng={d['lng']}"
            ),
        },
        "bolt_bike": {
            "app": f"bolt://ridepicker?type=bike&pickup={p['lat']},{p['lng']}&destination={d['lat']},{d['lng']}",
            "web": (
                f"https://bolt.eu/en/scooters/?pickup_latitude={p['lat']}&pickup_longitude={p['lng']}"
                f"&destination_latitude={d['lat']}&destination_longitude={d['lng']}"
            ),
        },
    }
    return links.get(platform, {"app": "", "web": ""})


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
@dataclass
class PlatformQuote:
    platform: str
    name: str
    type: str          # "car" | "motorbike"
    price_kes: int
    duration_min: int
    distance_km: float
    surge: bool
    source: str        # "ai" | "formula"
    deep_link_app: str = ""
    deep_link_web: str = ""
    is_cheapest: bool = False


async def get_all_quotes(
    pickup_address: str,
    dropoff_address: str,
    pickup_lat: float,
    pickup_lng: float,
    drop_lat: float,
    drop_lng: float,
) -> list[PlatformQuote]:
    """Return price quotes from all platforms, sorted cheapest first."""
    distance_km = _haversine(pickup_lat, pickup_lng, drop_lat, drop_lng)
    duration_min = _duration_min(distance_km)

    # Check cache first
    ckey = _cache_key(pickup_lat, pickup_lng, drop_lat, drop_lng)
    cached = _get_cache(ckey)

    if cached:
        raw_quotes = cached
        source_tag = "cache"
    else:
        # Try Gemini first, fall back to formula
        raw_quotes = await _call_gemini(
            pickup_address, dropoff_address,
            pickup_lat, pickup_lng, drop_lat, drop_lng,
            distance_km, duration_min,
        ) or _formula_prices(distance_km, duration_min)
        _set_cache(ckey, raw_quotes)

    quotes: list[PlatformQuote] = []
    for q in raw_quotes:
        links = get_deep_links(q["platform"], pickup_lat, pickup_lng, drop_lat, drop_lng)
        quotes.append(PlatformQuote(
            platform=q["platform"],
            name=q["name"],
            type=q["type"],
            price_kes=q["price_kes"],
            duration_min=q["duration_min"],
            distance_km=q["distance_km"],
            surge=q["surge"],
            source=q["source"],
            deep_link_app=links["app"],
            deep_link_web=links["web"],
        ))

    if quotes:
        min_price = min(q.price_kes for q in quotes)
        for q in quotes:
            q.is_cheapest = (q.price_kes == min_price)

    return quotes


def format_comparison_message(quotes: list[PlatformQuote], pickup: str, dropoff: str) -> str:
    """Format a WhatsApp-friendly comparison message with numbered choices."""
    if not quotes:
        return "Sorry, could not fetch prices right now. Please try again."

    lines = [
        f"🚖 *Price Comparison*",
        f"📍 {pickup} → {dropoff}",
        f"📏 {quotes[0].distance_km:.1f} km • ~{quotes[0].duration_min} min",
        "",
    ]

    cars = [q for q in quotes if q.type == "car"]
    bikes = [q for q in quotes if q.type == "motorbike"]

    if cars:
        lines.append("🚗 *Cars:*")
        for i, q in enumerate(cars, 1):
            badge = " 🏆 CHEAPEST" if q.is_cheapest and len(bikes) == 0 else (" 🏆" if q.is_cheapest else "")
            surge_tag = " ⚡surge" if q.surge else ""
            lines.append(f"  {i}. *{q.name}* — KES {q.price_kes}{surge_tag}{badge}")

    if bikes:
        lines.append("")
        lines.append("🏍️ *Boda Boda:*")
        offset = len(cars)
        for i, q in enumerate(bikes, 1):
            badge = " 🏆 CHEAPEST" if q.is_cheapest else ""
            surge_tag = " ⚡surge" if q.surge else ""
            lines.append(f"  {offset + i}. *{q.name}* — KES {q.price_kes}{surge_tag}{badge}")

    lines.append("")
    lines.append("Reply with the *number* to book (e.g. *1* for Uber)")
    lines.append("Or reply *BACK* to change destination")

    return "\n".join(lines)
