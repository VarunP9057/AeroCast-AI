import asyncio
from datetime import datetime, timezone
from urllib.request import Request, urlopen
from urllib.error import URLError, HTTPError
import json

from fastapi import APIRouter, HTTPException

router = APIRouter(
    prefix="/api/airport",
    tags=["Airport Live View"]
)

SFO_LATITUDE = 37.6213
SFO_LONGITUDE = -122.3790
SEARCH_RADIUS_NM = 25


def fetch_aircraft_data():
    url = (
        f"https://api.adsb.lol/v2/point/"
        f"{SFO_LATITUDE}/{SFO_LONGITUDE}/{SEARCH_RADIUS_NM}"
    )

    request = Request(
        url,
        headers={"User-Agent": "AeroCast-AI/1.0"}
    )

    with urlopen(request, timeout=15) as response:
        return json.loads(response.read().decode("utf-8"))


@router.get("/live")
async def get_live_aircraft():
    try:
        data = await asyncio.to_thread(fetch_aircraft_data)

        aircraft_list = []

        for aircraft in data.get("ac", []):
            latitude = aircraft.get("lat")
            longitude = aircraft.get("lon")

            if latitude is None or longitude is None:
                continue

            altitude = aircraft.get("alt_baro")

            aircraft_list.append({
                "hex": aircraft.get("hex"),
                "callsign": (aircraft.get("flight") or "").strip(),
                "registration": aircraft.get("r"),
                "aircraft_type": aircraft.get("t"),
                "latitude": latitude,
                "longitude": longitude,
                "altitude": altitude,
                "speed": aircraft.get("gs"),
                "heading": aircraft.get("track"),
                "vertical_rate": aircraft.get("baro_rate"),
                "squawk": aircraft.get("squawk"),
                "on_ground": altitude == "ground",
                "last_seen_seconds": aircraft.get("seen"),
                "distance_nm": aircraft.get("dst")
            })

        return {
            "airport": "San Francisco International Airport",
            "airport_code": "KSFO",
            "source": "ADSB.lol",
            "status": "live",
            "fetched_at": datetime.now(timezone.utc).isoformat(),
            "aircraft_count": len(aircraft_list),
            "aircraft": aircraft_list
        }

    except HTTPError as error:
        raise HTTPException(
            status_code=502,
            detail=f"Aircraft data provider returned HTTP {error.code}"
        )

    except (URLError, TimeoutError):
        raise HTTPException(
            status_code=503,
            detail="Live aircraft data is temporarily unavailable"
        )

    except (json.JSONDecodeError, ValueError):
        raise HTTPException(
            status_code=502,
            detail="Invalid response received from aircraft data provider"
        )
