from fastapi import APIRouter, HTTPException, Query

from backend.services.model_lab_service import (
    get_model_lab_overview,
    get_model_lab_results
)

router = APIRouter(
    prefix="/api/model-lab",
    tags=["Model Laboratory"]
)


@router.get("/overview")
def model_lab_overview():
    try:
        return get_model_lab_overview()
    except Exception as error:
        raise HTTPException(status_code=500, detail=str(error))


@router.get("/results")
def model_lab_results(dataset: str | None = Query(default=None)):
    try:
        return get_model_lab_results(dataset)
    except Exception as error:
        raise HTTPException(status_code=500, detail=str(error))
