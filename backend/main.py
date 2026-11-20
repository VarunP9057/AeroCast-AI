
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from backend.routes.assistant import router as assistant_router
from backend.routes.terminal_forecast import router as terminal_forecast_router
from backend.routes.aircraft_forecast import router as aircraft_forecast_router
from backend.routes.forecast import router as forecast_router
from backend.routes.analytics import router as analytics_router
from backend.routes.cargo_forecast import router as cargo_forecast_router
from backend.routes.operational_pressure_forecast import router as operational_pressure_router
from backend.routes.model_lab import router as model_lab_router
from backend.routes.airport_live import router as airport_live_router

app = FastAPI(
    title="AeroCast AI API",
    description="AI-Powered Airport Traffic Analytics and Passenger Forecasting System",
    version="1.1.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

app.include_router(forecast_router)
app.include_router(analytics_router)
app.include_router(terminal_forecast_router)
app.include_router(aircraft_forecast_router)
app.include_router(cargo_forecast_router)
app.include_router(operational_pressure_router)
app.include_router(model_lab_router)
app.include_router(airport_live_router)
app.include_router(assistant_router)

@app.get("/")
def home():
    return {
        "message": "Welcome to AeroCast AI Backend",
        "project": "AI-Powered Airport Traffic Analytics",
        "status": "Running"
    }


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "application": "AeroCast AI"
    }
