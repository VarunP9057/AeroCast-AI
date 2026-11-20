
from pydantic import BaseModel, Field


class ForecastRequest(BaseModel):
    Year: int = Field(..., ge=1999, le=2100)
    Month: int = Field(..., ge=1, le=12)
    Quarter: int = Field(..., ge=1, le=4)

    Month_Sin: float
    Month_Cos: float

    Lag_1: float = Field(..., ge=0)
    Lag_2: float = Field(..., ge=0)
    Lag_3: float = Field(..., ge=0)
    Lag_6: float = Field(..., ge=0)
    Lag_12: float = Field(..., ge=0)

    Rolling_Mean_3: float = Field(..., ge=0)
    Rolling_Mean_6: float = Field(..., ge=0)


class CustomForecastRequest(BaseModel):
    passenger_counts: list[float] = Field(
        ...,
        min_length=12,
        max_length=12,
        description="12 consecutive monthly passenger counts, oldest to newest"
    )
    target_year: int = Field(..., ge=1999, le=2100)
    target_month: int = Field(..., ge=1, le=12)


class ForecastResponse(BaseModel):
    predicted_passengers: float
    model_name: str
    status: str


class CustomForecastResponse(BaseModel):
    predicted_passengers: float
    target_month: str
    input_history: list[float]
    features_used: dict[str, float]
    model_name: str
    status: str
