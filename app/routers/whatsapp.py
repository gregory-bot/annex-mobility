"""Twilio WhatsApp webhook.

Endpoint: POST /webhooks/twilio/whatsapp
"""
from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Depends, Form, Response
from sqlalchemy.ext.asyncio import AsyncSession

from app.bot.engine import handle_message
from app.db import get_session

router = APIRouter(prefix="/webhooks/twilio", tags=["twilio"])


def _twiml(text: str) -> Response:
    body = f"<?xml version='1.0' encoding='UTF-8'?><Response><Message>{text}</Message></Response>"
    return Response(content=body, media_type="application/xml")


@router.post("/whatsapp")
async def whatsapp_webhook(
    From: str = Form(...),
    Body: str = Form(""),
    ButtonPayload: Optional[str] = Form(None),
    Latitude: Optional[float] = Form(None),
    Longitude: Optional[float] = Form(None),
    db: AsyncSession = Depends(get_session),
):
    phone = From.replace("whatsapp:", "").strip()

    user_message = Body
    if ButtonPayload:
        user_message = ButtonPayload

    reply = await handle_message(
        db, phone=phone, channel="whatsapp", body=user_message,
        latitude=Latitude, longitude=Longitude
    )

    return _twiml(reply.text)


@router.post("/sms")
async def twilio_sms_webhook(
    From: str = Form(...),
    Body: str = Form(""),
    db: AsyncSession = Depends(get_session),
):
    reply = await handle_message(db, phone=From.strip(), channel="sms", body=Body)
    return _twiml(reply.text)