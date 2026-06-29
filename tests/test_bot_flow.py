"""Integration tests for WaziRide v2 bot — full booking flow with platform comparison."""
import pytest

from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker

from app.db import Base
from app.bot.engine import handle_message
from app.services import drivers as drv_svc

TEST_DB = "sqlite+aiosqlite:///:memory:"


@pytest.fixture(scope="module")
def anyio_backend():
    return "asyncio"


@pytest.fixture(scope="module")
async def db_session():
    engine = create_async_engine(TEST_DB, echo=False)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    factory = async_sessionmaker(engine, expire_on_commit=False)
    async with factory() as session:
        await drv_svc.seed(session)
        yield session
    await engine.dispose()


async def send(db, phone, body, **kwargs):
    reply = await handle_message(db, phone=phone, channel="whatsapp", body=body, **kwargs)
    return reply.text


@pytest.mark.anyio
async def test_full_booking_flow(db_session):
    phone = "+254799000001"

    # 1. Greeting
    r = await send(db_session, phone, "hi")
    assert "Welcome" in r
    assert "pick you up" in r.lower()

    # 2. Pickup
    r = await send(db_session, phone, "Westlands, Nairobi")
    assert "Pickup" in r or "pickup" in r.lower()
    assert "Westlands" in r or "going" in r.lower()

    # 3. Destination → should show comparison table
    r = await send(db_session, phone, "JKIA, Nairobi")
    assert "Price Comparison" in r or "Bolt" in r or "Uber" in r
    assert "KES" in r
    assert "Reply with the number" in r or "number" in r.lower()

    # 4. Choose platform 1
    r = await send(db_session, phone, "1")
    assert "KES" in r
    assert "YES" in r or "confirm" in r.lower()

    # 5. Confirm
    r = await send(db_session, phone, "YES")
    assert "Driver" in r or "driver" in r.lower()
    assert "KES" in r


@pytest.mark.anyio
async def test_back_navigation(db_session):
    phone = "+254799000002"

    await send(db_session, phone, "hi")
    await send(db_session, phone, "Westlands, Nairobi")
    r = await send(db_session, phone, "Karen, Nairobi")
    assert "KES" in r  # comparison shown

    # Go back from platform selection
    r = await send(db_session, phone, "back")
    assert "going" in r.lower() or "destination" in r.lower()


@pytest.mark.anyio
async def test_no_then_yes(db_session):
    phone = "+254799000003"

    await send(db_session, phone, "hi")
    await send(db_session, phone, "-1.2921,36.8219")   # GPS coords
    await send(db_session, phone, "Gigiri, Nairobi")
    await send(db_session, phone, "2")  # choose platform 2

    # Say NO first — should go back to comparison
    r = await send(db_session, phone, "no")
    assert "KES" in r

    # Choose again and confirm
    await send(db_session, phone, "1")
    r = await send(db_session, phone, "YES")
    assert "Booked" in r or "Driver" in r


@pytest.mark.anyio
async def test_sos_and_cancel(db_session):
    phone = "+254799000004"

    await send(db_session, phone, "hi")
    await send(db_session, phone, "CBD, Nairobi")
    await send(db_session, phone, "Upperhill, Nairobi")
    await send(db_session, phone, "1")
    await send(db_session, phone, "YES")

    r = await send(db_session, phone, "SOS")
    assert "SOS" in r or "safety" in r.lower() or "999" in r

    r = await send(db_session, phone, "cancel")
    assert "cancel" in r.lower() or "HI" in r


@pytest.mark.anyio
async def test_help(db_session):
    phone = "+254799000005"
    r = await send(db_session, phone, "help")
    assert "HI" in r or "START" in r


@pytest.mark.anyio
async def test_status_no_trip(db_session):
    phone = "+254799000099"
    r = await send(db_session, phone, "status")
    assert "no active trip" in r.lower() or "HI" in r
