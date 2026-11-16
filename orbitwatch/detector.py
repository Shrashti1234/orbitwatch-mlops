import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score


WINDOW = 24
MIN_ALERT_LENGTH = 3
COMBINED_THRESHOLD = 1.0


def component_scores(features: pd.DataFrame, model, center: float) -> dict[str, np.ndarray]:
    return {
        "distance_from_median": (features["reading"] - center).abs().to_numpy(),
        "isolation_forest": -model.score_samples(features),
    }


def combined_scores(scores: dict[str, np.ndarray], thresholds: dict[str, float]) -> np.ndarray:
    return np.maximum.reduce([score / max(thresholds[name], 1e-6) for name, score in scores.items()])


def make_features(readings: np.ndarray) -> pd.DataFrame:
    series = pd.Series(readings, dtype=float)
    history = series.shift(1).rolling(WINDOW)
    return pd.DataFrame({
        "reading": series,
        "change": series.diff().abs(),
        "deviation": (series - history.median()).abs(),
        "prior_variability": history.std(ddof=0),
    }).dropna()


def alert_ranges(flags: np.ndarray, minimum: int = MIN_ALERT_LENGTH) -> list[tuple[int, int]]:
    edges = np.diff(np.r_[False, flags, False].astype(int))
    starts = np.flatnonzero(edges == 1)
    stops = np.flatnonzero(edges == -1)
    return [(int(start), int(stop - 1)) for start, stop in zip(starts, stops) if stop - start >= minimum]


def interval_mask(intervals: list[tuple[int, int]], count: int) -> np.ndarray:
    mask = np.zeros(count, dtype=bool)
    for start, end in intervals:
        mask[start:end + 1] = True
    return mask


def evaluate(scores: np.ndarray, threshold: float, labels: list[tuple[int, int]]) -> tuple[dict, list]:
    scored = np.isfinite(scores)
    truth = interval_mask(labels, len(scores))
    predicted = alert_ranges(scores > threshold)
    alerted = interval_mask(predicted, len(scores))
    detected_count = sum(bool(alerted[start:end + 1].any()) for start, end in labels)
    false_count = sum(not truth[start:end + 1].any() for start, end in predicted)
    metrics = {
        "point_average_precision": float(average_precision_score(truth[scored], scores[scored])),
        "events_detected": detected_count,
        "events_labeled": len(labels),
        "alert_intervals": len(predicted),
        "unmatched_alert_intervals": false_count,
        "alerted_samples": int(alerted.sum()),
        "alerted_samples_outside_labels": int((alerted & ~truth).sum()),
    }
    return metrics, predicted
