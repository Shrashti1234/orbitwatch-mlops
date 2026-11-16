import json
import os
from contextlib import asynccontextmanager
from functools import lru_cache
from pathlib import Path
from typing import Annotated, Literal

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, FiniteFloat

from .data import CHANNELS
from .detector import COMBINED_THRESHOLD, WINDOW, alert_ranges, combined_scores, component_scores, make_features


STATIC = Path(__file__).parent / "static"
Reading = Annotated[FiniteFloat, Field(ge=-1_000_000, le=1_000_000)]
Detector = Literal["isolation_forest", "distance_from_median", "combined"]


class ScoreRequest(BaseModel):
    readings: list[Reading] = Field(min_length=WINDOW + 1, max_length=10_000)


def nullable_scores(values) -> list[float | None]:
    return [float(value) if np.isfinite(value) else None for value in values]


def create_app(artifacts: Path | None = None) -> FastAPI:
    directory = artifacts or Path(os.environ.get("ORBITWATCH_ARTIFACTS", "artifacts"))

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        try:
            app.state.metrics = json.loads((directory / "metrics.json").read_text(encoding="utf-8"))["channels"]
            app.state.models = {channel: joblib.load(directory / f"{channel}-model.joblib") for channel in CHANNELS}
        except FileNotFoundError as error:
            raise RuntimeError("Generate the model artifacts with python -m orbitwatch.train before starting the API.") from error
        yield

    app = FastAPI(title="OrbitWatch", description="Score archived spacecraft telemetry for unusual behavior.", lifespan=lifespan)
    app.mount("/static", StaticFiles(directory=STATIC), name="static")

    @lru_cache(maxsize=len(CHANNELS) * 3)
    def replay(channel: str, detector: Detector) -> dict:
        table = pd.read_csv(directory / f"{channel}-timeline.csv")
        record = app.state.metrics[channel]
        metrics = record[detector]
        return {
            "channel": channel,
            "detector": detector,
            "readings": table["reading"].tolist(),
            "scores": nullable_scores(table[f"{detector}_score"]),
            "threshold": metrics["threshold"],
            "alerts": metrics["alerts"],
            "labeled_intervals": record["labeled_intervals"],
            "metrics": {field: value for field, value in metrics.items() if field not in {"threshold", "alerts"}},
        }

    def check_channel(channel: str) -> None:
        if channel not in CHANNELS:
            raise HTTPException(status_code=404, detail="Unknown SMAP channel")

    @app.get("/", include_in_schema=False)
    def dashboard():
        return FileResponse(STATIC / "index.html")

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok", "channels_loaded": len(app.state.models)}

    @app.get("/api/channels")
    def channels() -> dict:
        return {"channels": list(CHANNELS)}

    @app.get("/api/replay/{channel}")
    def archived_replay(channel: str, detector: Detector = "isolation_forest") -> dict:
        check_channel(channel)
        return replay(channel, detector)

    @app.post("/api/score/{channel}")
    def score(channel: str, request: ScoreRequest, detector: Detector = "isolation_forest") -> dict:
        check_channel(channel)
        readings = np.asarray(request.readings, dtype=float)
        features = make_features(readings)
        bundle = app.state.models[channel]
        scores = component_scores(features, bundle["model"], bundle["center"])
        if detector == "combined":
            scores["combined"] = combined_scores(scores, bundle["thresholds"])
        threshold = bundle["thresholds"].get(detector, COMBINED_THRESHOLD)
        aligned = np.full(len(readings), np.nan)
        aligned[features.index] = scores[detector]
        return {
            "channel": channel,
            "detector": detector,
            "scores": nullable_scores(aligned),
            "threshold": threshold,
            "alerts": alert_ranges(aligned > threshold),
        }

    return app


app = create_app()
