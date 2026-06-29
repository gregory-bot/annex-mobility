"""Conversational state machine - WhatsApp + SMS ride booking.

Design goals:
- No emojis. Plain text only.
- Any message starts a conversation (niaje, hello, sasa, etc.)
- Natural language understanding via Gemini
- Live location sharing after booking
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.entities import Driver, Session, Trip, User
from app.services import drivers as driver_svc
from app.services.geocoder import GeocodeResult, geocode
from app.services.ai_pricing import get_all_quotes, PlatformQuote
from app.services.nlu import extract_ride_intent

logger = logging.getLogger(__name__)


@dataclass
class Reply:
    text: str


HELP_TEXT = """
[Annex Mobility]

Available commands:
  HI / START - Begin a new booking
  STATUS     - Check your current trip
  LOCATION   - See driver's location
  CANCEL     - Cancel your ride
  SOS        - Emergency help
  HELP       - Show this menu

Or just tell me where you want to go. For example:
  "I need a ride from Westlands to JKIA"
  "Get me an Uber to Sarit Centre"
  "Bolt boda to CBD"

Tip: You can also share a WhatsApp location pin.
"""

WELCOME_MESSAGE = (
    "Welcome to Annex Mobility.\n\n"
    "I can help you book a ride from multiple providers "
    "(Uber, Bolt, Little, Faras, Yego) and find you the best price.\n\n"
    "Where would you like to go?\n\n"
    "For example: \"I need a ride from Westlands to JKIA\"\n"
    "Or just tell me your pickup point and I will guide you from there."
)

CASUAL_RESPONSES = {
    "niaje": "Niaje! Where would you like to go today?",
    "sasa": "Sasa! Ready to find you a ride. Where are you heading?",
    "habari": "Habari! Where can I take you today?",
    "mambo": "Mambo! Tell me where you need to go.",
    "poa": "Poa! Where would you like to go?",
    "vipi": "Vipi! Where are you heading today?",
    "hello": "Hello! Where would you like to go?",
    "hi": "Hi there! Where can I take you?",
    "hey": "Hey! Where are you heading?",
    "good morning": "Good morning! Where would you like to go today?",
    "good afternoon": "Good afternoon! Where can I take you?",
    "good evening": "Good evening! Where are you heading?",
    "how are you": "I am doing great, thanks for asking! How can I help you get where you need to go?",
    "how are you doing": "I am good! Ready to help you find a ride. Where are you heading?",
    "what's up": "Ready to help you get moving! Where would you like to go?",
    "sup": "Ready when you are! Where are you heading?",
}


# ---------------------------------------------------------------------------
# DB helpers
# ---------------------------------------------------------------------------

async def _get_or_create_user(db: AsyncSession, phone: str, channel: str) -> User:
    row = (await db.execute(select(User).where(User.phone == phone))).scalar_one_or_none()
    if row:
        if row.channel != channel:
            row.channel = channel
            await db.commit()
        return row
    user = User(phone=phone, channel=channel)
    db.add(user)
    await db.commit()
    await db.refresh(user)
    return user


async def _get_session(db: AsyncSession, user: User) -> Session:
    row = (await db.execute(select(Session).where(Session.user_id == user.id))).scalar_one_or_none()
    if row:
        return row
    s = Session(user_id=user.id, state="idle", data="{}")
    db.add(s)
    await db.commit()
    await db.refresh(s)
    return s


async def _set_state(db: AsyncSession, sess: Session, state: str, data: Optional[dict] = None) -> None:
    sess.state = state
    if data is not None:
        sess.data = json.dumps(data)
    await db.commit()


async def _active_trip(db: AsyncSession, user: User) -> Optional[Trip]:
    row = (
        await db.execute(
            select(Trip)
            .where(
                Trip.user_id == user.id,
                Trip.status.in_(["requested", "matched", "arrived", "in_progress"]),
            )
            .order_by(Trip.id.desc())
            .limit(1)
        )
    ).scalar_one_or_none()
    return row


def _parse_latlng(text: str) -> Optional[tuple[float, float]]:
    try:
        a, b = text.split(",", 1)
        lat, lng = float(a.strip()), float(b.strip())
        if -90 <= lat <= 90 and -180 <= lng <= 180:
            return lat, lng
    except Exception:
        pass
    return None


async def _resolve_location(text: str, lat: Optional[float] = None, lng: Optional[float] = None) -> Optional[GeocodeResult]:
    if lat is not None and lng is not None:
        addr = text.strip() or f"{lat:.5f},{lng:.5f}"
        return GeocodeResult(address=addr, lat=lat, lng=lng)
    ll = _parse_latlng(text)
    if ll:
        return GeocodeResult(address=f"({ll[0]:.5f}, {ll[1]:.5f})", lat=ll[0], lng=ll[1])
    return await geocode(text)


def _quotes_from_data(data: dict) -> list[dict]:
    return data.get("quotes", [])


def _platform_from_choice(quotes: list[dict], choice: str) -> Optional[dict]:
    """Map user reply to a platform quote. Numbers match display order: cars first, then bikes."""
    choice = choice.strip()

    # Build ordered list matching display: cars first, then bikes
    cars = [q for q in quotes if q["type"] == "car"]
    bikes = [q for q in quotes if q["type"] == "motorbike"]
    ordered = cars + bikes

    # Numeric choice
    if choice.isdigit():
        idx = int(choice) - 1
        if 0 <= idx < len(ordered):
            return ordered[idx]

    # Name match (uber, bolt, little, faras, yego, bolt_bike)
    for q in quotes:
        if q["platform"].lower() == choice.lower() or q["name"].lower() == choice.lower():
            return q

    return None


# ---------------------------------------------------------------------------
# Message formatting (no emojis, plain text)
# ---------------------------------------------------------------------------

def _format_comparison(quotes: list[PlatformQuote], pickup: str, dropoff: str) -> str:
    """Format platform comparison as clean text with numbered choices."""
    if not quotes:
        return "Sorry, could not fetch prices right now. Please try again."

    cars = [q for q in quotes if q.type == "car"]
    bikes = [q for q in quotes if q.type == "motorbike"]

    lines = [
        "--- RIDE OPTIONS ---",
        f"From: {pickup}",
        f"To: {dropoff}",
        f"Distance: {quotes[0].distance_km:.1f} km",
        f"Est. time: ~{quotes[0].duration_min} min",
        "",
    ]

    if cars:
        lines.append("CARS:")
        for i, q in enumerate(cars, 1):
            badge = " [CHEAPEST]" if q.is_cheapest and len(bikes) == 0 else ""
            surge = " (surge)" if q.surge else ""
            lines.append(f"  {i}. {q.name} - KES {q.price_kes}{surge}{badge}")

    if bikes:
        lines.append("")
        lines.append("BODA BODA:")
        offset = len(cars)
        for i, q in enumerate(bikes, 1):
            badge = " [CHEAPEST]" if q.is_cheapest else ""
            surge = " (surge)" if q.surge else ""
            lines.append(f"  {offset + i}. {q.name} - KES {q.price_kes}{surge}{badge}")

    lines.append("")
    lines.append("Reply with the number of your choice (e.g. 1)")
    lines.append("Or type: uber, bolt, cheapest")
    lines.append("Reply BACK to change destination")

    return "\n".join(lines)


def _format_booking_confirmation(chosen: dict, driver: Driver) -> str:
    """Format booking confirmation."""
    web_link = chosen.get("deep_link_web", "")
    platform_name = chosen.get("name", "Ride")

    msg = f"""
--- BOOKING CONFIRMED ---

Platform: {platform_name}
Driver: {driver.name}
Vehicle: {driver.vehicle}
Plate: {driver.plate}
Rating: {driver.rating}

Fare: KES {chosen['price_kes']}
Est. time: ~{chosen['duration_min']} min
Distance: {chosen['distance_km']:.1f} km

Pickup: {chosen.get('pickup_text', '')}
Dropoff: {chosen.get('dropoff_text', '')}
"""

    if web_link:
        msg += f"\nTrack your ride: {web_link}"

    msg += """
---
You can reply:
  STATUS - Check trip status
  LOCATION - See driver position on map
  CANCEL - Cancel ride
  SOS - Emergency
"""
    return msg.strip()


def _format_driver_location(driver: Driver, trip: Trip, loc_text: str) -> str:
    """Format driver location with Google Maps link."""
    maps_url = f"https://www.google.com/maps?q={trip.pickup_lat},{trip.pickup_lng}"
    return f"""
--- DRIVER LOCATION ---

Driver: {driver.name}
Vehicle: {driver.vehicle} ({driver.plate})
Current area: {loc_text}
Phone: {driver.phone}

View on map: {maps_url}

For your safety, share this with a friend.
""".strip()


def _format_trip_status(trip: Trip, driver: Optional[Driver] = None) -> str:
    """Format trip status message."""
    status_labels = {
        "requested": "Looking for driver...",
        "matched": "Driver confirmed",
        "arrived": "Driver has arrived",
        "in_progress": "Trip in progress",
    }
    label = status_labels.get(trip.status, trip.status)

    maps_url = f"https://www.google.com/maps?q={trip.pickup_lat},{trip.pickup_lng}"

    lines = [
        f"--- TRIP #{trip.id} ---",
        f"Status: {label}",
        f"Platform: {trip.chosen_platform_name or 'N/A'}",
        f"From: {trip.pickup_text}",
        f"To: {trip.dropoff_text}",
        f"Fare: KES {trip.fare_kes:.0f}",
        f"Distance: {trip.distance_km:.1f} km",
        f"Est. time: ~{trip.duration_min:.0f} min",
    ]

    if driver:
        lines.append("")
        lines.append(f"Driver: {driver.name}")
        lines.append(f"Vehicle: {driver.vehicle} ({driver.plate})")
        lines.append(f"Rating: {driver.rating}")
        lines.append(f"Phone: {driver.phone}")

    now = datetime.now(timezone.utc)
    if trip.created_at:
        elapsed = (now - trip.created_at.replace(tzinfo=timezone.utc)).total_seconds() / 60
        eta = max(trip.duration_min - elapsed, 1)
        lines.append(f"ETA: ~{eta:.0f} min remaining")

    lines.append(f"\nView on map: {maps_url}")

    return "\n".join(lines)


def _format_sos(trip: Trip, driver: Optional[Driver], user_phone: str) -> str:
    """Format SOS emergency alert."""
    lines = [
        "--- EMERGENCY - SOS ACTIVATED ---",
        "",
        f"Passenger: {user_phone}",
        f"Trip #{trip.id}",
        f"Pickup: {trip.pickup_text}",
        f"Dropoff: {trip.dropoff_text}",
        f"Fare: KES {trip.fare_kes:.0f}",
        f"Platform: {trip.chosen_platform_name or 'N/A'}",
        f"Time: {trip.created_at}",
        "",
    ]
    if driver:
        lines.append(f"Driver: {driver.name}")
        lines.append(f"Vehicle: {driver.vehicle} ({driver.plate})")
        lines.append(f"Phone: {driver.phone}")
    else:
        lines.append("Driver: Not yet assigned")

    lines.append("")
    lines.append("Our safety team has been notified.")
    lines.append("If in danger, call 999 immediately.")
    lines.append("Police: 999 | Ambulance: 112")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# NLU fast path
# ---------------------------------------------------------------------------

async def _try_nlu_fast_path(
    db: AsyncSession,
    user: User,
    sess: Session,
    body: str,
) -> Optional[Reply]:
    """If NLU extracts both pickup and dropoff, fast-forward to pricing."""
    data = json.loads(sess.data or "{}")

    context = {}
    if data.get("pickup"):
        context["pickup"] = data["pickup"]["text"]
    if data.get("dropoff"):
        context["dropoff"] = data["dropoff"]["text"]

    extracted = await extract_ride_intent(body, context)
    if not extracted:
        return None

    if extracted.pickup and extracted.dropoff:
        loc_pickup = await _resolve_location(extracted.pickup)
        loc_dropoff = await _resolve_location(extracted.dropoff)

        if not loc_pickup:
            return Reply(
                f"I found the destination ({extracted.dropoff}) but could not locate "
                "the pickup. Can you clarify where you are?"
            )
        if not loc_dropoff:
            return Reply(
                f"I found your pickup ({extracted.pickup}) but could not locate "
                "the destination. Where are you heading?"
            )

        new_data = {
            "pickup": {"text": loc_pickup.address, "lat": loc_pickup.lat, "lng": loc_pickup.lng},
            "dropoff": {"text": loc_dropoff.address, "lat": loc_dropoff.lat, "lng": loc_dropoff.lng},
            "platform_preference": extracted.platform_preference,
            "vehicle_type": extracted.vehicle_type,
        }

        try:
            quotes = await get_all_quotes(
                pickup_address=loc_pickup.address,
                dropoff_address=loc_dropoff.address,
                pickup_lat=loc_pickup.lat,
                pickup_lng=loc_pickup.lng,
                drop_lat=loc_dropoff.lat,
                drop_lng=loc_dropoff.lng,
            )
        except Exception as exc:
            logger.error("Price fetch failed: %s", exc)
            return Reply("Could not fetch prices right now. Please try again. Send HI to restart.")

        if not quotes:
            return Reply("Could not fetch prices right now. Please try again. Send HI to restart.")

        if extracted.platform_preference and extracted.platform_preference != "cheapest":
            filtered = [q for q in quotes if q.platform == extracted.platform_preference]
            if filtered:
                quotes = filtered

        if extracted.vehicle_type == "motorbike":
            filtered = [q for q in quotes if q.type == "motorbike"]
            if filtered:
                quotes = filtered
        elif extracted.vehicle_type == "car":
            filtered = [q for q in quotes if q.type == "car"]
            if filtered:
                quotes = filtered

        quotes_json = [
            {
                "platform": q.platform, "name": q.name, "type": q.type,
                "price_kes": q.price_kes, "duration_min": q.duration_min,
                "distance_km": q.distance_km, "surge": q.surge, "source": q.source,
                "deep_link_app": q.deep_link_app, "deep_link_web": q.deep_link_web,
                "is_cheapest": q.is_cheapest,
            }
            for q in quotes
        ]

        new_data["quotes"] = quotes_json
        await _set_state(db, sess, "awaiting_platform", new_data)

        pref_note = ""
        if extracted.platform_preference and extracted.platform_preference != "cheapest":
            pref_note = f"\n(Filtered for: {extracted.platform_preference})"
        elif extracted.vehicle_type:
            pref_note = f"\n(Filtered for: {extracted.vehicle_type})"

        comparison = _format_comparison(quotes, loc_pickup.address, loc_dropoff.address)
        return Reply(comparison + pref_note)

    if extracted.pickup and not extracted.dropoff and not data.get("pickup"):
        loc = await _resolve_location(extracted.pickup)
        if not loc:
            return Reply("Sorry, I could not find that location. Please try a clearer address.")
        await _set_state(db, sess, "awaiting_dropoff", {
            "pickup": {"text": loc.address, "lat": loc.lat, "lng": loc.lng}
        })
        return Reply(
            f"Pickup set to: {loc.address}\n\n"
            "Where are you going?\n"
            "(Send an address or share a location pin)"
        )

    return None


# ---------------------------------------------------------------------------
# Main handler
# ---------------------------------------------------------------------------

async def handle_message(
    db: AsyncSession,
    *,
    phone: str,
    channel: str,
    body: str,
    latitude: Optional[float] = None,
    longitude: Optional[float] = None,
) -> Reply:
    body_raw = (body or "").strip()
    body_lower = body_raw.lower()

    user = await _get_or_create_user(db, phone, channel)
    sess = await _get_session(db, user)

    # Global commands
    if body_lower in {"help", "menu"}:
        return Reply(HELP_TEXT.strip())

    if body_lower == "sos":
        trip = await _active_trip(db, user)
        if trip:
            driver = None
            if trip.driver_id:
                driver = (await db.execute(
                    select(Driver).where(Driver.id == trip.driver_id)
                )).scalar_one_or_none()
            alert = _format_sos(trip, driver, phone)
            logger.critical("SOS activated: trip=%s user=%s", trip.id, phone)
            return Reply(alert)
        return Reply("You do not have an active trip. If this is an emergency, call 999 immediately.")

    if body_lower == "cancel":
        trip = await _active_trip(db, user)
        if trip:
            trip.status = "cancelled"
            if trip.driver_id:
                await driver_svc.release(db, trip.driver_id)
            await db.commit()
        await _set_state(db, sess, "idle", {})
        return Reply("Trip cancelled.\n\nWhere would you like to go?\n(Send an address or share a location pin)")

    # In-trip
    if sess.state == "in_trip":
        trip = await _active_trip(db, user)
        if not trip:
            await _set_state(db, sess, "idle", {})
            return Reply("Your trip has ended. Send HI to book another ride.")

        driver = None
        if trip.driver_id:
            driver = (await db.execute(
                select(Driver).where(Driver.id == trip.driver_id)
            )).scalar_one_or_none()

        if body_lower in {"status", "update"}:
            return Reply(_format_trip_status(trip, driver))

        if body_lower in {"location", "where", "position"}:
            if driver:
                from app.services.geocoder import reverse_geocode
                loc_text = await reverse_geocode(trip.pickup_lat, trip.pickup_lng)
                return Reply(_format_driver_location(driver, trip, loc_text))
            return Reply("Driver location not available yet. Reply STATUS for trip details.")

        if body_lower in {"arrived", "picked", "start"}:
            trip.status = "in_progress"
            await db.commit()
            return Reply("Trip started. Safe journey! Reply STATUS anytime, or DONE when you arrive.")

        if body_lower in {"done", "complete", "completed", "finish"}:
            trip.status = "completed"
            trip.completed_at = datetime.utcnow()
            if trip.driver_id:
                await driver_svc.release(db, trip.driver_id)
            await db.commit()
            await _set_state(db, sess, "idle", {})
            return Reply(
                f"Trip #{trip.id} completed.\nFare: KES {trip.fare_kes:.0f}\n\n"
                "Thank you for riding with Annex Mobility.\nSend HI to book another trip."
            )

        return Reply("You have an active trip.\nReply: STATUS | LOCATION | ARRIVED | DONE | CANCEL | SOS")

    # NLU fast path
    if sess.state in {"idle", "awaiting_pickup"}:
        nlu_reply = await _try_nlu_fast_path(db, user, sess, body_raw)
        if nlu_reply:
            return nlu_reply

    # idle
    if sess.state == "idle":
        casual = CASUAL_RESPONSES.get(body_lower)
        if casual:
            await _set_state(db, sess, "awaiting_pickup", {})
            return Reply(casual)

        if body_lower in {"ride", "book", "start", "go"}:
            await _set_state(db, sess, "awaiting_pickup", {})
            return Reply(WELCOME_MESSAGE)

        nlu_reply = await _try_nlu_fast_path(db, user, sess, body_raw)
        if nlu_reply:
            return nlu_reply

        await _set_state(db, sess, "awaiting_pickup", {})
        loc = await _resolve_location(body_raw, latitude, longitude)
        if loc:
            await _set_state(db, sess, "awaiting_dropoff", {
                "pickup": {"text": loc.address, "lat": loc.lat, "lng": loc.lng}
            })
            return Reply(f"Pickup: {loc.address}\n\nWhere are you going?\n(Send an address or share a location pin)")
        return Reply(WELCOME_MESSAGE)

    # awaiting_pickup
    if sess.state == "awaiting_pickup":
        loc = await _resolve_location(body_raw, latitude, longitude)
        if not loc:
            return Reply("Sorry, I could not find that location. Please try a clearer address or share a location pin.")
        await _set_state(db, sess, "awaiting_dropoff", {
            "pickup": {"text": loc.address, "lat": loc.lat, "lng": loc.lng}
        })
        return Reply(f"Pickup: {loc.address}\n\nWhere are you going?\n(Send an address or share a destination pin)")

    # awaiting_dropoff
    if sess.state == "awaiting_dropoff":
        if body_lower in {"back", "change"}:
            await _set_state(db, sess, "awaiting_pickup", {})
            return Reply("Where should we pick you up?")

        loc = await _resolve_location(body_raw, latitude, longitude)
        if not loc:
            return Reply("Sorry, I could not find that destination. Try a clearer address.")

        data = json.loads(sess.data or "{}")
        p = data.get("pickup")
        if not p:
            await _set_state(db, sess, "idle", {})
            return Reply("Something went wrong. Send HI to start again.")

        try:
            quotes = await get_all_quotes(
                pickup_address=p["text"], dropoff_address=loc.address,
                pickup_lat=p["lat"], pickup_lng=p["lng"],
                drop_lat=loc.lat, drop_lng=loc.lng,
            )
        except Exception as exc:
            logger.error("Price fetch failed: %s", exc)
            return Reply("Could not fetch prices right now. Please try again. Send HI to restart.")

        if not quotes:
            return Reply("Could not fetch prices right now. Please try again. Send HI to restart.")

        quotes_json = [
            {
                "platform": q.platform, "name": q.name, "type": q.type,
                "price_kes": q.price_kes, "duration_min": q.duration_min,
                "distance_km": q.distance_km, "surge": q.surge, "source": q.source,
                "deep_link_app": q.deep_link_app, "deep_link_web": q.deep_link_web,
                "is_cheapest": q.is_cheapest,
            }
            for q in quotes
        ]

        data["dropoff"] = {"text": loc.address, "lat": loc.lat, "lng": loc.lng}
        data["quotes"] = quotes_json
        await _set_state(db, sess, "awaiting_platform", data)

        return Reply(_format_comparison(quotes, p["text"], loc.address))

    # awaiting_platform
    if sess.state == "awaiting_platform":
        if body_lower in {"back", "change"}:
            data = json.loads(sess.data or "{}")
            p = data.get("pickup", {})
            await _set_state(db, sess, "awaiting_dropoff", {"pickup": p})
            return Reply("Where are you going?")

        data = json.loads(sess.data or "{}")
        quotes = _quotes_from_data(data)

        if body_lower in {"cheapest", "lowest", "best price"}:
            cars = [q for q in quotes if q["type"] == "car"]
            bikes = [q for q in quotes if q["type"] == "motorbike"]
            ordered = cars + bikes
            chosen = ordered[0] if ordered else None
        else:
            chosen = _platform_from_choice(quotes, body_lower)

        if not chosen:
            total = len(quotes)
            return Reply(
                f"Please reply with a number between 1 and {total} to choose.\n"
                f"Or type: cheapest, uber, bolt, etc.\nReply BACK to change destination."
            )

        p = data.get("pickup", {})
        d = data.get("dropoff", {})
        chosen["pickup_text"] = p.get("text", "")
        chosen["dropoff_text"] = d.get("text", "")

        data["chosen"] = chosen
        await _set_state(db, sess, "awaiting_confirm", data)

        surge_note = " (includes surge pricing)" if chosen.get("surge") else ""

        return Reply(
            f"--- YOUR SELECTION ---\n\n"
            f"Platform: {chosen['name']}{surge_note}\n"
            f"From: {p.get('text', '')}\n"
            f"To: {d.get('text', '')}\n"
            f"Fare: KES {chosen['price_kes']}\n"
            f"Est. time: ~{chosen['duration_min']} min\n"
            f"Distance: {chosen['distance_km']:.1f} km\n\n"
            "Reply YES to confirm or NO to choose a different option."
        )

    # awaiting_confirm
    if sess.state == "awaiting_confirm":
        if body_lower in {"no", "n", "back", "change"}:
            data = json.loads(sess.data or "{}")
            quotes_raw = _quotes_from_data(data)
            p = data.get("pickup", {})
            d = data.get("dropoff", {})
            if quotes_raw:
                quotes_obj = [
                    PlatformQuote(
                        platform=q["platform"], name=q["name"], type=q["type"],
                        price_kes=q["price_kes"], duration_min=q["duration_min"],
                        distance_km=q["distance_km"], surge=q["surge"], source=q["source"],
                        deep_link_app=q.get("deep_link_app", ""),
                        deep_link_web=q.get("deep_link_web", ""),
                        is_cheapest=q.get("is_cheapest", False),
                    )
                    for q in quotes_raw
                ]
                data.pop("chosen", None)
                await _set_state(db, sess, "awaiting_platform", data)
                return Reply(_format_comparison(quotes_obj, p.get("text", ""), d.get("text", "")))
            await _set_state(db, sess, "idle", {})
            return Reply("Cancelled. Send HI to start again.")

        if body_lower in {"yes", "y", "confirm", "ok"}:
            data = json.loads(sess.data or "{}")
            p = data.get("pickup")
            d = data.get("dropoff")
            chosen = data.get("chosen")

            if not (p and d and chosen):
                await _set_state(db, sess, "idle", {})
                return Reply("Something went wrong. Send HI to start again.")

            trip = Trip(
                user_id=user.id,
                pickup_text=p["text"], pickup_lat=p["lat"], pickup_lng=p["lng"],
                dropoff_text=d["text"], dropoff_lat=d["lat"], dropoff_lng=d["lng"],
                distance_km=chosen["distance_km"],
                duration_min=chosen["duration_min"],
                fare_kes=float(chosen["price_kes"]),
                chosen_platform=chosen["platform"],
                chosen_platform_name=chosen["name"],
                status="requested",
            )
            db.add(trip)
            await db.commit()
            await db.refresh(trip)

            drv = await driver_svc.find_available(db)
            if not drv:
                trip.status = "cancelled"
                await db.commit()
                await _set_state(db, sess, "idle", {})
                return Reply("No drivers available right now.\nPlease try again in a few minutes. Send HI to retry.")
            trip.driver_id = drv.id
            trip.status = "matched"
            await db.commit()

            await _set_state(db, sess, "in_trip", {"trip_id": trip.id})
            return Reply(_format_booking_confirmation(chosen, drv))

        return Reply("Please reply YES to confirm or NO to choose a different option.")

    # Fallback
    await _set_state(db, sess, "idle", {})
    return Reply("Send a message to start a booking, or HELP for commands.")