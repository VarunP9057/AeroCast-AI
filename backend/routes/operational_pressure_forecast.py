
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from backend.services.operational_pressure_ann_service import (
    predict_latest_operational_pressure_ann,
    predict_custom_operational_pressure_ann
)

from backend.services.operational_pressure_service import (
    predict_latest_operational_pressure,
    predict_operational_pressure
)

router = APIRouter(
    prefix="/api/forecast/operational-pressure",
    tags=["Operational Pressure Forecasting"]
)


class OperationalPressureRequest(BaseModel):
    features: dict[str, float]


class CustomOperationalPressureANNRequest(BaseModel):
    passenger_counts: list[float]
    landing_counts: list[float]
    cargo_weights: list[float]
    target_year: int
    target_month: int
@router.get("/")
def operational_pressure_info():
    return {
        "message": "Operational Pressure Forecasting API",
        "model": "Operational Pressure Linear Regression",
        "status": "active"
    }


@router.get("/latest")
def latest_operational_pressure():
    try:
        return predict_latest_operational_pressure()
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )


@router.post("/custom")
def custom_operational_pressure(request: OperationalPressureRequest):
    try:
        prediction = predict_operational_pressure(
            request.features
        )

        return {
            "predicted_operational_pressure_index": prediction,
            "model_name": "Operational Pressure Linear Regression",
            "status": "success"
        }

    except ValueError as e:
        raise HTTPException(
            status_code=400,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )
@router.get("/ann/latest")
def latest_operational_pressure_ann():
    try:
        return predict_latest_operational_pressure_ann()

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
    
@router.post("/ann/custom-history")
def custom_operational_pressure_ann(
    request: CustomOperationalPressureANNRequest
):
    try:
        return predict_custom_operational_pressure_ann(
            passenger_counts=request.passenger_counts,
            landing_counts=request.landing_counts,
            cargo_weights=request.cargo_weights,
            target_year=request.target_year,
            target_month=request.target_month
        )

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
