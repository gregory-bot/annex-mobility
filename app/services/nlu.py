"""Natural Language Understanding via Gemini - extracts pickup, dropoff, and ride preference from free-text messages.

Allows users to say things like:
  "I need a ride from Westlands to JKIA, the cheapest option"
  "Get me an Uber to Sarit Centre"
  "Bolt bike to Ngong Road, leaving from Kilimani"
and the bot extracts structured data in one shot.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Optional

import httpx

from app.config import settings

logger = logging.getLogger(__name__)


@dataclass
class ExtractedRide:
    pickup: Optional[str] = None
    dropoff: Optional[str] = None
    platform_preference: Optional[str] = None
    vehicle_type: Optional[str] = None
    confidence: float = 0.0


async def extract_ride_intent(user_message: str, context: Optional[dict] = None) -> Optional[ExtractedRide]:
    """Use Gemini to extract pickup, dropoff, and preferences from a natural-language message.
    
    Returns None if Gemini is unavailable or the message is not ride-related.
    """
    if not settings.GEMINI_API_KEY:
        return None

    context_str = ""
    if context:
        ctx_parts = []
        if context.get("pickup"):
            ctx_parts.append(f"Known pickup: {context['pickup']}")
        if context.get("dropoff"):
            ctx_parts.append(f"Known dropoff: {context['dropoff']}")
        context_str = chr(92) + "n".join(ctx_parts)

    numbered_list = chr(92) + "n".join([
        "1. pickup_location - the pickup point",
        "2. dropoff_location - the destination", 
        "3. platform_preference - uber, bolt, little, faras, yego, bolt_bike, cheapest, fastest, or null",
        "4. vehicle_type - car, motorbike, or null",
        "5. is_ride_request - true if booking a ride, false otherwise"
    ])

    prompt = f"""You are a ride-hailing NLU system for Kenya. Extract structured data from user messages.

Context (already known):
{context_str or "None yet - this is the start of a conversation"}

User message: "{user_message}"

Determine:
{numbered_list}

Rules:
- If user says hi, hello, help, status, cancel, sos, yes, no, or a number: is_ride_request=false
- If both pickup and dropoff are mentioned, extract both
- If only one location, put it in pickup_location
- cheapest means platform_preference=cheapest
- boda, bike, motorbike means vehicle_type=motorbike

Return ONLY valid JSON:
{{"pickup_location": "string or null", "dropoff_location": "string or null", "platform_preference": "string or null", "vehicle_type": "string or null", "is_ride_request": true/false}}"""

    url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent"
    headers = {"Content-Type": "application/json"}
    params = {"key": settings.GEMINI_API_KEY}
    body = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0.0, "maxOutputTokens": 256},
    }

    try:
        async with httpx.AsyncClient(timeout=8) as client:
            r = await client.post(url, headers=headers, params=params, json=body)
            r.raise_for_status()
            data = r.json()

        text = data["candidates"][0]["content"]["parts"][0]["text"]
        text = text.strip().lstrip("`json").lstrip("`").rstrip("`").strip()
        parsed = json.loads(text)

        if not parsed.get("is_ride_request"):
            return None

        return ExtractedRide(
            pickup=parsed.get("pickup_location"),
            dropoff=parsed.get("dropoff_location"),
            platform_preference=parsed.get("platform_preference"),
            vehicle_type=parsed.get("vehicle_type"),
            confidence=0.9,
        )

    except Exception as exc:
        logger.debug("NLU extraction skipped: %s", exc)
        return None
