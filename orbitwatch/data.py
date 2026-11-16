import ast
from pathlib import Path
from shutil import copyfileobj
from urllib.request import urlopen

import numpy as np
import pandas as pd


CHANNELS = tuple(f"A-{number}" for number in range(1, 10))
MIRROR = (
    "https://huggingface.co/datasets/appleparan/telemanom/resolve/"
    "2d22e1061be83a88b7b9e48df35163d5147adc9d/data"
)
LABELS = (
    "https://raw.githubusercontent.com/khundman/telemanom/"
    "2e6c5b6c3558e7835601519b7bdef37c649bdbdc/labeled_anomalies.csv"
)


def download(url: str, destination: Path) -> Path:
    if destination.is_file():
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    with urlopen(url, timeout=45) as response, temporary.open("wb") as output:
        copyfileobj(response, output)
    temporary.replace(destination)
    return destination


def load_telemetry(cache: Path, split: str, channel: str) -> np.ndarray:
    path = download(f"{MIRROR}/{split}/{channel}.parquet", cache / split / f"{channel}.parquet")
    table = pd.read_parquet(path, columns=["timestep", "value"])
    if not np.array_equal(table["timestep"], np.arange(len(table))):
        raise ValueError(f"{split}/{channel} does not have contiguous zero-based timesteps")
    values = table["value"].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError(f"{split}/{channel} contains non-finite readings")
    return values


def load_labels(cache: Path, channel: str, count: int) -> list[tuple[int, int]]:
    labels = pd.read_csv(download(LABELS, cache / "labeled_anomalies.csv")).set_index("chan_id")
    record = labels.loc[channel]
    if record["spacecraft"] != "SMAP" or int(record["num_values"]) != count:
        raise ValueError(f"Label metadata does not match SMAP test channel {channel}")
    intervals = [tuple(pair) for pair in ast.literal_eval(record["anomaly_sequences"])]
    if any(not 0 <= start <= end < count for start, end in intervals):
        raise ValueError(f"Label interval outside test channel {channel}")
    return sorted(intervals)
