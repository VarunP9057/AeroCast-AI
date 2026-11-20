
from fastapi import APIRouter, HTTPException

from backend.services.cargo_forecast_service import (
    predict_cargo_weight,
    predict_latest_cargo_weight,
    predict_custom_cargo_weight
)

router = APIRouter(
    prefix="/api/forecast/cargo",
    tags=["Cargo Forecasting"]
)


@router.get("/")
def cargo_forecast_info():
    return {
        "message": "Cargo Weight Forecasting API",
        "model": "Cargo Advanced ANN - Experiment 2",
        "features": 24,
        "unit": "lbs",
        "status": "active"
    }


@router.get("/latest")
def latest_cargo_forecast():
    try:
        return predict_latest_cargo_weight()

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
def custom_cargo_forecast(features: dict):
    try:
        prediction = predict_cargo_weight(features)

        return {
            "predicted_cargo_weight_lbs": prediction,
            "unit": "lbs",
            "model_name": "Cargo Advanced ANN - Experiment 2",
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
def custom_cargo_history_forecast(request: dict):
    try:
        result = predict_custom_cargo_weight(
            cargo_weights=request.get("cargo_weights", []),
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

