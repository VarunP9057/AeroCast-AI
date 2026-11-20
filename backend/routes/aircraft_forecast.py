from backend.services.aircraft_forecast_service import (
    predict_aircraft_landings,
    predict_latest_aircraft_landings,
    predict_custom_aircraft_landings
)
from fastapi import APIRouter, HTTPException

from backend.services.aircraft_forecast_service import (
    predict_aircraft_landings,
    predict_latest_aircraft_landings
)

router = APIRouter(
    prefix="/api/forecast/aircraft",
    tags=["Aircraft Forecasting"]
)


@router.get("/")
def aircraft_forecast_info():
    return {
        "message": "Aircraft Landing Forecasting API",
        "model": "Original Aircraft ANN",
        "features": 14,
        "status": "active"
    }


@router.get("/latest")
def latest_aircraft_forecast():
    try:
        return predict_latest_aircraft_landings()

    except (ValueError, FileNotFoundError) as error:
        raise HTTPException(
            status_code=400,
            detail=str(error)
        )

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=str(error)
        )


@router.post("/custom")
def custom_aircraft_forecast(features: dict):
    try:
        prediction = predict_aircraft_landings(features)

        return {
            "predicted_landings": prediction,
            "model_name": "Original Aircraft ANN",
            "status": "success"
        }

    except ValueError as error:
        raise HTTPException(
            status_code=400,
            detail=str(error)
        )

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=str(error)
        )

@router.post("/custom-history")
def custom_aircraft_history_forecast(request: dict):
    try:
        result = predict_custom_aircraft_landings(
            landing_counts=request.get("landing_counts", []),
            target_year=request.get("target_year"),
            target_month=request.get("target_month")
        )

        return result

    except ValueError as error:
        raise HTTPException(
            status_code=400,
            detail=str(error)
        )

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=str(error)
        )

