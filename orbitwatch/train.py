import argparse
import json
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from .data import CHANNELS, load_labels, load_telemetry
from .detector import (
    COMBINED_THRESHOLD, MIN_ALERT_LENGTH, WINDOW, combined_scores, component_scores,
    evaluate, interval_mask, make_features,
)


FIT_FRACTION = 0.75
ALERT_QUANTILE = 0.99


def run_channel(channel: str, cache: Path, output: Path) -> dict:
    train = make_features(load_telemetry(cache, "train", channel))
    readings = load_telemetry(cache, "test", channel)
    test = make_features(readings)
    labels = load_labels(cache, channel, len(readings))
    split = int(len(train) * FIT_FRACTION)
    fit, calibration = train.iloc[:split], train.iloc[split:]

    center = float(fit["reading"].median())
    model = IsolationForest(n_estimators=150, max_samples=256, random_state=17)
    model.fit(fit)
    calibration_scores = component_scores(calibration, model, center)
    thresholds = {name: float(np.quantile(scores, ALERT_QUANTILE)) for name, scores in calibration_scores.items()}
    test_scores = component_scores(test, model, center)
    test_scores["combined"] = combined_scores(test_scores, thresholds)
    report = {}
    timeseries = pd.DataFrame({"timestep": np.arange(len(readings)), "reading": readings})
    timeseries["labeled_anomaly"] = interval_mask(labels, len(readings))
    for name, scores in test_scores.items():
        threshold = thresholds.get(name, COMBINED_THRESHOLD)
        aligned = np.full(len(readings), np.nan)
        aligned[test.index] = scores
        metrics, alerts = evaluate(aligned, threshold, labels)
        report[name] = {"threshold": threshold, **metrics, "alerts": alerts}
        timeseries[f"{name}_score"] = aligned
        timeseries[f"{name}_alert"] = interval_mask(alerts, len(readings))

    timeseries.to_csv(output / f"{channel}-timeline.csv", index=False)
    joblib.dump({
        "channel": channel,
        "model": model,
        "center": center,
        "thresholds": thresholds,
        "lookback": WINDOW,
        "minimum_alert_length": MIN_ALERT_LENGTH,
    }, output / f"{channel}-model.joblib")
    return {
        "training_rows": len(fit),
        "calibration_rows": len(calibration),
        "test_rows_scored": len(test),
        "labeled_intervals": labels,
        **report,
    }


def track_run(uri: str, output: Path, report: dict) -> None:
    import mlflow

    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment("OrbitWatch SMAP")
    with mlflow.start_run(run_name="smap-a-channel-backtest") as run:
        mlflow.log_params({
            "fit_fraction": FIT_FRACTION,
            "calibration_quantile": ALERT_QUANTILE,
            "lookback_samples": WINDOW,
            "min_alert_samples": MIN_ALERT_LENGTH,
            "channels": ",".join(CHANNELS),
        })
        for detector in ("distance_from_median", "isolation_forest", "combined"):
            summary = [result[detector] for result in report["channels"].values()]
            for field in ("events_detected", "events_labeled", "unmatched_alert_intervals", "alerted_samples_outside_labels"):
                mlflow.log_metric(f"{detector}_{field}", sum(item[field] for item in summary))
        for channel, result in report["channels"].items():
            mlflow.log_metric(f"{channel}_average_precision", result["combined"]["point_average_precision"])
            mlflow.log_artifact(str(output / f"{channel}-model.joblib"), artifact_path="models")
        mlflow.log_artifact(str(output / "metrics.json"))
        print(f"MLflow run ID: {run.info.run_id}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Backtest detectors on historical NASA SMAP telemetry.")
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--output", type=Path, default=Path("artifacts"))
    parser.add_argument("--tracking-uri", default=os.environ.get("MLFLOW_TRACKING_URI"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    results = {channel: run_channel(channel, args.data, args.output) for channel in CHANNELS}
    report = {"lookback": WINDOW, "calibration_quantile": ALERT_QUANTILE, "channels": results}
    (args.output / "metrics.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    for channel, result in results.items():
        for name in ("distance_from_median", "isolation_forest", "combined"):
            stats = result[name]
            print(f"{channel} {name}: AP {stats['point_average_precision']:.3f}, "
                  f"detected {stats['events_detected']}/{stats['events_labeled']} events, "
                  f"{stats['unmatched_alert_intervals']} unmatched alerts")
    if args.tracking_uri:
        track_run(args.tracking_uri, args.output, report)


if __name__ == "__main__":
    main()
