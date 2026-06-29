"""Conversation state machine — WhatsApp + SMS ride booking with multi-platform comparison.

States:
  idle                -> greeting, prompt for pickup
  awaiting_pickup     -> store pickup, ask for dropoff
  awaiting_dropoff    -> store dropoff, fetch AI quotes, show comparison
  awaiting_platform   -> user picks a platform number
  awaiting_confirm    -> show chosen platform summary, ask YES/NO
  in_trip             -> handles STATUS / ARRIVED / DONE / SOS / CANCEL

New flow (after dropoff is entered):
  1. Bot fetches prices from Gemini AI for all platforms
  2. Presents numbered list: e.g. "1. Uber – KES 850, 2. Bolt – KES 720 ..."
  3. User replies with a number (e.g. "2")
  4. Bot confirms: "You chose Bolt (KES 720). Reply YES to confirm."
  5. User replies YES → trip created, driver assigned
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.entities import Driver, Session, Trip, User
from app.services import drivers as driver_svc
from app.services.geocoder import GeocodeResult, geocode
from app.services.ai_pricing import get_all_quotes, format_comparison_message, PlatformQuote

logger = logging.getLogger(__name__)


@dataclass
class Reply:
    text: str


HELP = (
    "📋 *Commands*\n"
    "• HI / START — begin a new booking\n"
    "• STATUS — current trip status\n"
    "• CANCEL — cancel current request\n"
    "• SOS — emergency help\n"
    "• HELP — show this menu"
)


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
    # Explicit GPS pin (from WhatsApp location share)
    if lat is not None and lng is not None:
        addr = text.strip() or f"{lat:.5f},{lng:.5f}"
        return GeocodeResult(address=addr, lat=lat, lng=lng)
    # Raw "lat,lng" text
    ll = _parse_latlng(text)
    if ll:
        return GeocodeResult(address=f"({ll[0]:.5f}, {ll[1]:.5f})", lat=ll[0], lng=ll[1])
    return await geocode(text)


def _quotes_from_data(data: dict) -> list[dict]:
    """Restore quotes list from session JSON."""
    return data.get("quotes", [])


def _platform_from_choice(quotes: list[dict], choice: str) -> Optional[dict]:
    """Map user reply (number or name) to a platform quote."""
    choice = choice.strip()
    # Numeric choice
    if choice.isdigit():
        idx = int(choice) - 1
        if 0 <= idx < len(quotes):
            return quotes[idx]
    # Name match (e.g. "uber", "bolt")
    for q in quotes:
        if q["platform"].lower() == choice.lower() or q["name"].lower() == choice.lower():
            return q
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

    # ------------------------------------------------------------------ #
    # Global commands (work from any state)
    # ------------------------------------------------------------------ #
    if body_lower in {"help", "menu", "?"}:
        return Reply(HELP)

    if body_lower == "sos":
        trip = await _active_trip(db, user)
        ctx = f" Trip #{trip.id}, driver #{trip.driver_id}." if trip else ""
        return Reply(
            "🆘 *SOS received.* Our safety team has been notified."
            + ctx
            + "\nIf life is in danger, call *999* immediately."
        )

    if body_lower == "cancel":
        trip = await _active_trip(db, user)
        if trip:
            trip.status = "cancelled"
            if trip.driver_id:
                await driver_svc.release(db, trip.driver_id)
            await db.commit()
        await _set_state(db, sess, "idle", {})
        return Reply("❌ Cancelled. Send *HI* to start a new booking.")

    if body_lower == "status":
        trip = await _active_trip(db, user)
        if not trip:
            return Reply("You have no active trip. Send *HI* to book a ride.")
        drv = await db.get(Driver, trip.driver_id) if trip.driver_id else None
        platform_tag = f" via *{trip.chosen_platform_name}*" if trip.chosen_platform_name else ""
        msg = (
            f"📋 Trip #{trip.id}{platform_tag} — *{trip.status.upper()}*\n"
            f"From: {trip.pickup_text}\nTo: {trip.dropoff_text}\n"
            f"💰 KES {trip.fare_kes:.0f} • {trip.distance_km:.1f} km • ~{trip.duration_min:.0f} min"
        )
        if drv:
            msg += f"\n🚗 Driver: {drv.name} ({drv.plate}) ⭐ {drv.rating}\n📞 {drv.phone}"
        return Reply(msg)

    # ------------------------------------------------------------------ #
    # Greeting / reset
    # ------------------------------------------------------------------ #
    if body_lower in {"hi", "hello", "start", "hey", "habari", "hallo"} or (
        sess.state == "idle" and body_lower not in {"no", "n"}
    ):
        if body_lower in {"hi", "hello", "start", "hey", "habari", "hallo"}:
            await _set_state(db, sess, "awaiting_pickup", {})
            return Reply(
                "👋 *Welcome to Annex Mobility!*\n\n"
                "We compare Uber, Bolt, Little, Faras & Yego to find you the best price.\n\n"
                "📍 Where should we pick you up?\n"
                "_(Send an address, share a location pin, or type lat,lng)_"
            )

    # ------------------------------------------------------------------ #
    # awaiting_pickup
    # ------------------------------------------------------------------ #
    if sess.state == "awaiting_pickup":
        loc = await _resolve_location(body_raw, latitude, longitude)
        if not loc:
            return Reply("Sorry, I couldn't find that location. Please try a clearer address.")
        await _set_state(db, sess, "awaiting_dropoff", {
            "pickup": {"text": loc.address, "lat": loc.lat, "lng": loc.lng}
        })
        return Reply(
            f"✅ Pickup: *{loc.address}*\n\n"
            "🏁 Where are you going?\n"
            "_(Send an address or share a destination pin)_"
        )

    # ------------------------------------------------------------------ #
    # awaiting_dropoff → fetch AI prices, show comparison
    # ------------------------------------------------------------------ #
    if sess.state == "awaiting_dropoff":
        # Allow "back" to re-enter pickup
        if body_lower in {"back", "change"}:
            await _set_state(db, sess, "awaiting_pickup", {})
            return Reply("📍 Where should we pick you up?")

        loc = await _resolve_location(body_raw, latitude, longitude)
        if not loc:
            return Reply("Sorry, I couldn't find that destination. Try a clearer address.")

        data = json.loads(sess.data or "{}")
        p = data.get("pickup")
        if not p:
            await _set_state(db, sess, "idle", {})
            return Reply("Something went wrong. Send *HI* to start again.")

        # Fetch prices — this may call Gemini
        try:
            quotes = await get_all_quotes(
                pickup_address=p["text"],
                dropoff_address=loc.address,
                pickup_lat=p["lat"],
                pickup_lng=p["lng"],
                drop_lat=loc.lat,
                drop_lng=loc.lng,
            )
        except Exception as exc:
            logger.error("Price fetch failed: %s", exc)
            quotes = []

        if not quotes:
            return Reply(
                "⚠️ Could not fetch prices right now. Please try again in a moment.\n"
                "Or send *HI* to restart."
            )

        # Serialise quotes to JSON for session storage
        quotes_json = [
            {
                "platform": q.platform,
                "name": q.name,
                "type": q.type,
                "price_kes": q.price_kes,
                "duration_min": q.duration_min,
                "distance_km": q.distance_km,
                "surge": q.surge,
                "source": q.source,
                "deep_link_app": q.deep_link_app,
                "deep_link_web": q.deep_link_web,
                "is_cheapest": q.is_cheapest,
            }
            for q in quotes
        ]

        data["dropoff"] = {"text": loc.address, "lat": loc.lat, "lng": loc.lng}
        data["quotes"] = quotes_json
        await _set_state(db, sess, "awaiting_platform", data)

        comparison = format_comparison_message(quotes, p["text"], loc.address)
        return Reply(comparison)

    # ------------------------------------------------------------------ #
    # awaiting_platform — user picks a number
    # ------------------------------------------------------------------ #
    if sess.state == "awaiting_platform":
        if body_lower in {"back", "change"}:
            data = json.loads(sess.data or "{}")
            p = data.get("pickup", {})
            await _set_state(db, sess, "awaiting_dropoff", {"pickup": p})
            return Reply("🏁 Where are you going?")

        data = json.loads(sess.data or "{}")
        quotes = _quotes_from_data(data)
        chosen = _platform_from_choice(quotes, body_lower)

        if not chosen:
            total = len(quotes)
            return Reply(
                f"Please reply with a number between 1 and {total} to choose a platform.\n"
                f"Or type the name (e.g. *bolt*, *uber*).\n"
                f"Reply *BACK* to change destination."
            )

        # Store chosen platform in session
        data["chosen"] = chosen
        await _set_state(db, sess, "awaiting_confirm", data)

        p = data.get("pickup", {})
        d = data.get("dropoff", {})
        surge_note = " _(includes surge pricing)_" if chosen.get("surge") else ""
        booking_note = (
            f"\n\n📱 After booking, your {chosen['name']} app will open automatically."
            if chosen.get("deep_link_web") else ""
        )

        return Reply(
            f"✅ *{chosen['name']}* selected{surge_note}\n\n"
            f"📍 From: {p.get('text', '')}\n"
            f"📍 To: {d.get('text', '')}\n"
            f"💰 Fare: *KES {chosen['price_kes']}*\n"
            f"⏱ ~{chosen['duration_min']} min • {chosen['distance_km']:.1f} km"
            f"{booking_note}\n\n"
            "Reply *YES* to confirm or *NO* to choose a different platform."
        )

    # ------------------------------------------------------------------ #
    # awaiting_confirm — YES / NO
    # ------------------------------------------------------------------ #
    if sess.state == "awaiting_confirm":
        if body_lower in {"no", "n", "back", "change"}:
            # Go back to platform selection — re-show comparison
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
                return Reply(format_comparison_message(quotes_obj, p.get("text", ""), d.get("text", "")))
            await _set_state(db, sess, "idle", {})
            return Reply("Cancelled. Send *HI* to start again.")

        if body_lower in {"yes", "y", "confirm", "ok"}:
            data = json.loads(sess.data or "{}")
            p = data.get("pickup")
            d = data.get("dropoff")
            chosen = data.get("chosen")

            if not (p and d and chosen):
                await _set_state(db, sess, "idle", {})
                return Reply("Something went wrong. Send *HI* to start again.")

            # Create trip record
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

            # Match a driver
            drv = await driver_svc.find_available(db)
            if not drv:
                trip.status = "cancelled"
                await db.commit()
                await _set_state(db, sess, "idle", {})
                return Reply(
                    "😔 No drivers available right now.\n"
                    "Please try again in a few minutes. Send *HI* to retry."
                )
            trip.driver_id = drv.id
            trip.status = "matched"
            await db.commit()

            await _set_state(db, sess, "in_trip", {"trip_id": trip.id})

            # Deep link for user
            web_link = chosen.get("deep_link_web", "")
            app_launch = f"\n\n📱 *Open {chosen['name']}:* {web_link}" if web_link else ""

            return Reply(
                f"🎉 *Booked via {chosen['name']}!*\n\n"
                f"🚗 Driver: *{drv.name}*\n"
                f"🔢 Plate: {drv.plate}\n"
                f"🚙 {drv.vehicle}\n"
                f"⭐ Rating: {drv.rating}\n"
                f"📞 {drv.phone}\n\n"
                f"💰 Fare: *KES {trip.fare_kes:.0f}*\n"
                f"⏱ ~{chosen['duration_min']} min away"
                f"{app_launch}\n\n"
                "Reply *STATUS* for updates | *CANCEL* to cancel | *SOS* for emergency"
            )

        return Reply("Please reply *YES* to confirm or *NO* to choose a different platform.")

    # ------------------------------------------------------------------ #
    # in_trip
    # ------------------------------------------------------------------ #
    if sess.state == "in_trip":
        trip = await _active_trip(db, user)
        if not trip:
            await _set_state(db, sess, "idle", {})
            return Reply("Your trip has ended. Send *HI* to book another ride.")

        if body_lower in {"arrived", "picked", "start"}:
            trip.status = "in_progress"
            await db.commit()
            return Reply("🚦 Trip started. Safe journey! 🙏\nReply *DONE* when you arrive.")

        if body_lower in {"done", "complete", "completed", "finish"}:
            trip.status = "completed"
            trip.completed_at = datetime.utcnow()
            if trip.driver_id:
                await driver_svc.release(db, trip.driver_id)
            await db.commit()
            await _set_state(db, sess, "idle", {})
            platform_tag = f" via {trip.chosen_platform_name}" if trip.chosen_platform_name else ""
            return Reply(
                f"🏁 Trip #{trip.id}{platform_tag} completed!\n"
                f"💰 Fare: KES {trip.fare_kes:.0f}\n\n"
                "Thanks for riding with *Annex Mobility*! 🙏\n"
                "Send *HI* for another trip."
            )

        return Reply(
            "You're on a trip!\n"
            "Reply: *STATUS* | *ARRIVED* | *DONE* | *CANCEL* | *SOS*"
        )

    # ------------------------------------------------------------------ #
    # Fallback
    # ------------------------------------------------------------------ #
    await _set_state(db, sess, "idle", {})
    return Reply("Send *HI* to start a booking, or *HELP* for commands.")
