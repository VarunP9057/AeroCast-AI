
from fastapi import APIRouter, HTTPException

from backend.services.analytics_service import (
    get_passenger_analytics,
    get_detailed_passenger_analytics,
    get_aircraft_analytics,
    get_aircraft_monthly_analytics,
    get_cargo_analytics,
    get_cargo_monthly_analytics,
    get_integrated_analytics
)

router = APIRouter(
    prefix="/api/analytics",
    tags=["Airport Analytics"]
)


@router.get("/passengers")
def passenger_analytics():
    try:
        return get_passenger_analytics()
    except Exception as error:
        raise HTTPException(status_code=500, detail=str(error))


@router.get("/passengers/detailed")
def detailed_passenger_analytics():
    try:
        return get_detailed_passenger_analytics()
    except Exception as error:
        raise HTTPException(status_code=500, detail=str(error))


@router.get("/aircraft")
def aircraft_analytics():
    try:
        return get_aircraft_analytics()
    except Exception as error:
        raise HTTPException(status_code=500, detail=str(error))


@router.get("/aircraft/monthly")
def aircraft_monthly_analytics():
    try:
        return get_aircraft_monthly_analytics()
    except Exception as error:
        raise HTTPException(status_code=500, detail=str(error))


@router.get("/cargo")
def cargo_analytics():
    try:
        return get_cargo_analytics()
    except Exception as error:
        raise HTTPException(status_code=500, detail=str(error))


@router.get("/cargo/monthly")
def cargo_monthly_analytics():
    try:
        return get_cargo_monthly_analytics()
    except Exception as error:
        raise HTTPException(status_code=500, detail=str(error))


@router.get("/integrated")
def integrated_analytics():
    try:
        return get_integrated_analytics()
    except Exception as error:
        raise HTTPException(status_code=500, detail=str(error))
