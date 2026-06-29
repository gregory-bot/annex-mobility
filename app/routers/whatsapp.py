"""Twilio WhatsApp webhook with interactive button support.

Endpoint: POST /webhooks/twilio/whatsapp
"""
from __future__ import annotations

import json
from typing import Optional

from fastapi import APIRouter, Depends, Form, Response
from sqlalchemy.ext.asyncio import AsyncSession

from app.bot.engine import handle_message
from app.db import get_session

router = APIRouter(prefix="/webhooks/twilio", tags=["twilio"])


def _format_buttons_message(body_text: str, buttons: list[dict]) -> str:
    """Build a TwiML response with interactive buttons.
    Max 3 buttons per WhatsApp message, titles max 20 chars.
    """
    button_lines = []
    for btn in buttons[:3]:
        title = btn["title"][:20]
        button_lines.append(f'<Button id="{btn["id"]}">{title}</Button>')
    buttons_xml = "\n".join(button_lines)

    return f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
  <Message>
    <Body>{body_text}</Body>
    <Buttons>
      {buttons_xml}
    </Buttons>
  </Message>
</Response>"""


def _twiml(text: str) -> Response:
    """Plain text TwiML response."""
    body = f"<?xml version='1.0' encoding='UTF-8'?><Response><Message>{text}</Message></Response>"
    return Response(content=body, media_type="application/xml")


def _extract_buttons_from_text(text: str) -> Optional[tuple[str, list[dict]]]:
    """Extract [[BUTTONS:json]] marker. Returns (clean_text, buttons) or None."""
    marker = "[[BUTTONS:"
    if marker not in text:
        return None
    start = text.index(marker) + len(marker)
    end = text.index("]]", start)
    buttons_json = text[start:end]
    buttons = json.loads(buttons_json)
    clean_text = text.replace(f"{marker}{buttons_json}]]", "").strip()
    return clean_text, buttons


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

    # Button clicks come as ButtonPayload
    user_message = Body
    if ButtonPayload:
        user_message = ButtonPayload

    reply = await handle_message(
        db, phone=phone, channel="whatsapp", body=user_message,
        latitude=Latitude, longitude=Longitude
    )

    # Render buttons if marker present
    result = _extract_buttons_from_text(reply.text)
    if result:
        clean_text, buttons = result
        return Response(
            content=_format_buttons_message(clean_text, buttons),
            media_type="application/xml"
        )

    return _twiml(reply.text)


@router.post("/sms")
async def twilio_sms_webhook(
    From: str = Form(...),
    Body: str = Form(""),
    db: AsyncSession = Depends(get_session),
):
    reply = await handle_message(db, phone=From.strip(), channel="sms", body=Body)
    # SMS doesn't support buttons, strip markers
    clean = reply.text
    if "[[BUTTONS:" in clean:
        start = clean.index("[[BUTTONS:")
        end = clean.index("]]", start) + 2
        clean = clean[:start].strip()
    return _twiml(clean)