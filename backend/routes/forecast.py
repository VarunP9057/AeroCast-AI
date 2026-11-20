
from fastapi import APIRouter, HTTPException

from backend.schemas.forecast_schema import (
    ForecastRequest,
    ForecastResponse,
    CustomForecastRequest,
    CustomForecastResponse
)

from backend.services.forecast_service import (
    predict_passengers,
    predict_custom_passengers
)


router = APIRouter(
    prefix="/api",
    tags=["Passenger Forecasting"]
)


# EXISTING AUTOMATIC FORECAST
@router.post("/forecast", response_model=ForecastResponse)
def forecast_passengers(request: ForecastRequest):

    try:
        prediction = predict_passengers(request.model_dump())

        return ForecastResponse(
            predicted_passengers=prediction,
            model_name="Artificial Neural Network",
            status="success"
        )

    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error))

    except FileNotFoundError as error:
        raise HTTPException(status_code=500, detail=str(error))


# NEW CUSTOM DATA PREDICTION
@router.post(
    "/forecast/custom",
    response_model=CustomForecastResponse
)
def custom_forecast_passengers(request: CustomForecastRequest):

    try:
        result = predict_custom_passengers(
            passenger_counts=request.passenger_counts,
            target_year=request.target_year,
            target_month=request.target_month
        )

        return CustomForecastResponse(
            predicted_passengers=result["predicted_passengers"],
            target_month=(
                f"{request.target_year}-{request.target_month:02d}"
            ),
            input_history=request.passenger_counts,
            features_used=result["features_used"],
            model_name="Artificial Neural Network",
            status="success"
        )

    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error))

    except FileNotFoundError as error:
        raise HTTPException(status_code=500, detail=str(error))
