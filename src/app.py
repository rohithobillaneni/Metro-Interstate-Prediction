"""Local API for an explicitly trusted, newly trained Metro model bundle."""

from contextlib import asynccontextmanager
import math
import os
from pathlib import Path
from typing import Annotated, Literal

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

from src.artifacts import load_bundle
from src.offline_data import feature_values, hour_category


WEEKDAYS = ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday")
MONTHS = (
    "January", "February", "March", "April", "May", "June", "July", "August",
    "September", "October", "November", "December",
)


class TrafficData(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")

    temp: Annotated[float, Field(allow_inf_nan=False, gt=-273.15)]
    clouds_all: Annotated[float, Field(allow_inf_nan=False, ge=0, le=100)]
    holiday: Literal["No", "Yes"]
    weather_main: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=64)]
    weekday: Literal["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    hour: Annotated[int, Field(ge=0, le=23)]
    hour_category: Literal["Early Morning", "Morning", "Afternoon", "Evening", "Night", "Late Night"] | None = None
    month: Literal["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]

    @model_validator(mode="after")
    def consistent_hour_category(self):
        if self.hour_category is not None and self.hour_category != hour_category(self.hour):
            raise ValueError("hour_category contradicts hour; omit it to derive the category")
        return self

    def features(self) -> dict:
        return feature_values(
            temp=self.temp,
            clouds_all=self.clouds_all,
            holiday=int(self.holiday == "Yes"),
            weather_main=self.weather_main,
            weekday=WEEKDAYS.index(self.weekday),
            hour=self.hour,
            month=MONTHS.index(self.month) + 1,
        )


def create_app(
    model_directory: str | Path | None = None,
    trusted_manifest_sha256: str | None = None,
) -> FastAPI:
    """Import safely; configure and verify a trusted bundle at lifespan startup."""
    @asynccontextmanager
    async def lifespan(application: FastAPI):
        directory = model_directory if model_directory is not None else os.environ.get("METRO_MODEL_DIR")
        pin = trusted_manifest_sha256 if trusted_manifest_sha256 is not None else os.environ.get("METRO_MODEL_MANIFEST_SHA256")
        if not directory or not pin:
            raise RuntimeError("METRO_MODEL_DIR and separately trusted METRO_MODEL_MANIFEST_SHA256 are required")
        application.state.model = load_bundle(directory, pin)
        try:
            yield
        finally:
            del application.state.model

    application = FastAPI(lifespan=lifespan)

    @application.exception_handler(RequestValidationError)
    async def validation_error(request: Request, exc: RequestValidationError):
        # Do not echo input/ctx: NaN, Infinity and ValueError objects are not JSON
        # serializable, and reflecting arbitrary request data is unnecessary.
        details = [
            {"loc": list(error["loc"]), "type": error["type"], "msg": error["msg"]}
            for error in exc.errors()
        ]
        return JSONResponse(status_code=422, content={"detail": details})

    @application.post("/predict")
    def predict_traffic(data: TrafficData, request: Request):
        # A sync endpoint places CPU work in FastAPI's worker pool.
        try:
            prediction = float(request.app.state.model.predict(pd.DataFrame([data.features()]))[0])
            if not math.isfinite(prediction) or prediction < 0:
                raise ValueError("Model returned invalid traffic volume")
        except Exception as exc:
            # HTTP boundary: retain server exception context, never return paths,
            # request contents or deserialization details to the client.
            raise HTTPException(status_code=500, detail="Prediction failed") from exc
        return {"prediction": prediction}

    @application.get("/")
    def read_root():
        return {"message": "Welcome to the Metro Traffic Volume Prediction API!"}

    return application


app = create_app()
