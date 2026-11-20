
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from backend.services.terminal_forecast_service import (
    predict_latest_terminal_passengers,
    predict_terminal_passengers,
    get_all_terminal_predictions,
    predict_custom_terminal_passengers
)

router = APIRouter(
    prefix="/api/forecast/terminals",
    tags=["Terminal Passenger Forecasting"]
)


class CustomTerminalANNRequest(BaseModel):
    passenger_counts: list[float]
    target_year: int
    target_month: int


@router.get("/")
def terminal_forecast_home():
    return {
        "message": "Terminal Passenger Forecasting API",
        "supported_terminals": [
            "International",
            "Terminal 1",
            "Terminal 2",
            "Terminal 3"
        ],
        "status": "active"
    }


@router.get("/all")
def forecast_all_terminals():
    try:
        return get_all_terminal_predictions()

    except (ValueError, FileNotFoundError) as error:
        raise HTTPException(
            status_code=400 if isinstance(error, ValueError) else 500,
            detail=str(error)
        )

    except Exception as error:
        raise HTTPException(status_code=500, detail=str(error))


@router.post("/ann/custom-history/{terminal}")
def forecast_custom_terminal_ann(
    terminal: str,
    request: CustomTerminalANNRequest
):
    try:
        return predict_custom_terminal_passengers(
            terminal=terminal,
            passenger_counts=request.passenger_counts,
            target_year=request.target_year,
            target_month=request.target_month
        )

    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error))

    except FileNotFoundError as error:
        raise HTTPException(status_code=500, detail=str(error))

    except Exception as error:
        raise HTTPException(status_code=500, detail=str(error))


@router.get("/{terminal}")
def forecast_single_terminal(terminal: str):
    try:
        return predict_latest_terminal_passengers(terminal)

    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error))

    except FileNotFoundError as error:
        raise HTTPException(status_code=500, detail=str(error))

    except Exception as error:
        raise HTTPException(status_code=500, detail=str(error))


@router.post("/{terminal}/custom")
def forecast_custom_terminal(terminal: str, features: dict):
    try:
        return predict_terminal_passengers(
            terminal=terminal,
            features=features
        )

    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error))

    except FileNotFoundError as error:
        raise HTTPException(status_code=500, detail=str(error))

    except Exception as error:
        raise HTTPException(status_code=500, detail=str(error))
