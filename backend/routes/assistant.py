
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional

from backend.services.analytics_service import (
    get_passenger_analytics,
    get_aircraft_analytics,
    get_cargo_analytics,
    get_integrated_analytics,
    get_detailed_passenger_analytics,
)

router = APIRouter(prefix="/api/assistant", tags=["AeroCast AI Assistant"])


class ChatRequest(BaseModel):
    message: str


class ChatResponse(BaseModel):
    answer: str
    source: str = "AeroCast AI Analytics"


def format_number(value):
    return f"{value:,.0f}"


def generate_answer(question):
    q = question.lower()

    try:
        if any(word in q for word in ["airline", "carrier"]):
            data = get_detailed_passenger_analytics()
            airlines = data["airline_rankings"]

            if not airlines:
                return "Airline ranking data is unavailable."

            top = airlines[0]
            return (
                f"The highest passenger traffic is recorded by "
                f"{top['airline']}, with "
                f"{format_number(top['passengers'])} passengers."
            )

        if any(word in q for word in ["terminal", "boarding area"]):
            data = get_detailed_passenger_analytics()

            if "boarding" in q:
                items = data["boarding_area_distribution"]
                label = "boarding area"
                name_key = "boarding_area"
            else:
                items = data["terminal_distribution"]
                label = "terminal"
                name_key = "terminal"

            if not items:
                return f"{label.title()} data is unavailable."

            top = items[0]
            return (
                f"The {label} with the highest passenger traffic is "
                f"{top[name_key]}, with "
                f"{format_number(top['passengers'])} passengers."
            )

        if any(word in q for word in ["domestic", "international"]):
            data = get_detailed_passenger_analytics()
            items = data["domestic_international"]

            return "\n".join(
                f"{item['category']}: {format_number(item['passengers'])} passengers"
                for item in items
            )

        if any(word in q for word in ["peak month", "busiest month", "highest month"]):
            data = get_detailed_passenger_analytics()
            peak = data["peak_month"]

            return (
                f"SFO's peak passenger month in the historical dataset is "
                f"{peak['month']}, with "
                f"{format_number(peak['passengers'])} passengers."
            )

        if any(word in q for word in ["lowest month", "quietest month"]):
            data = get_detailed_passenger_analytics()
            lowest = data["lowest_month"]

            return (
                f"The lowest passenger traffic was recorded in "
                f"{lowest['month']}, with "
                f"{format_number(lowest['passengers'])} passengers."
            )

        if any(word in q for word in ["cargo", "freight"]):
            data = get_cargo_analytics()

            return (
                f"SFO has handled a total historical cargo weight of "
                f"{format_number(data['total_cargo_weight_lbs'])} lbs. "
                f"The latest recorded month is {data['latest_month']}, "
                f"with {format_number(data['latest_cargo_weight_lbs'])} lbs."
            )

        if any(word in q for word in ["aircraft", "landing", "landings"]):
            data = get_aircraft_analytics()

            return (
                f"SFO has recorded {format_number(data['total_landings'])} "
                f"aircraft landings in the historical dataset. "
                f"The latest recorded month is {data['latest_month']}, "
                f"with {format_number(data['latest_landings'])} landings."
            )

        if any(word in q for word in ["operational", "integrated", "airport activity"]):
            data = get_integrated_analytics()

            return (
                f"The integrated airport dataset contains "
                f"{format_number(data['total_passengers'])} passengers, "
                f"{format_number(data['total_landings'])} aircraft landings "
                f"and {format_number(data['total_cargo_weight_lbs'])} lbs "
                f"of cargo. Latest recorded month: {data['latest_month']}."
            )

        if any(word in q for word in ["average passenger", "monthly average"]):
            data = get_passenger_analytics()

            return (
                f"The average monthly passenger traffic is "
                f"{format_number(data['average_monthly_passengers'])} passengers."
            )

        if any(word in q for word in ["passenger", "traffic", "traveller", "travelers"]):
            data = get_passenger_analytics()

            return (
                f"SFO has recorded a total of "
                f"{format_number(data['total_passengers'])} passengers "
                f"across {data['total_records']} monthly records. "
                f"The latest recorded month is {data['latest_month']}, "
                f"with {format_number(data['latest_passengers'])} passengers."
            )

        return (
            "I can help you explore SFO passenger traffic, airline rankings, "
            "terminal distribution, aircraft landings, cargo statistics "
            "and historical airport activity. Please ask a question about "
            "one of these AeroCast AI modules."
        )

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Unable to retrieve airport analytics: {str(exc)}"
        )


@router.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    if not request.message.strip():
        raise HTTPException(status_code=400, detail="Please enter a question.")

    answer = generate_answer(request.message)

    return ChatResponse(answer=answer)
