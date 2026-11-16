import json
import math
import sys
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


def check(base: str) -> None:
    def call(path: str, readings: list[float] | None = None):
        body = json.dumps({"readings": readings}).encode() if readings is not None else None
        request = Request(base + path, data=body, headers={"Content-Type": "application/json"})
        with urlopen(request, timeout=8) as response:
            return json.load(response)

    deadline = time.monotonic() + 75
    while True:
        try:
            ready = call("/health")
            break
        except (URLError, ConnectionError, TimeoutError):
            if time.monotonic() >= deadline:
                raise
            time.sleep(1)
    assert ready == {"status": "ok", "channels_loaded": 9}, ready
    assert call("/api/channels")["channels"] == [f"A-{number}" for number in range(1, 10)]
    for detector in ("isolation_forest", "distance_from_median", "combined"):
        replay = call(f"/api/replay/A-3?detector={detector}")
        fresh = call(f"/api/score/A-3?detector={detector}", replay["readings"])
        assert fresh["threshold"] == replay["threshold"]
        assert fresh["alerts"] == replay["alerts"]
        assert len(fresh["scores"]) == len(replay["scores"])
        assert all(
            left is right if left is None or right is None else math.isclose(left, right, rel_tol=1e-9, abs_tol=1e-12)
            for left, right in zip(replay["scores"], fresh["scores"])
        )
    for path, readings, status in (("/api/replay/A-99", None, 404), ("/api/score/A-3", [0.1], 422)):
        try:
            call(path, readings)
        except HTTPError as error:
            assert error.code == status, error.code
        else:
            raise AssertionError(f"Expected HTTP {status} from {path}")
    with urlopen(base + "/", timeout=8) as response:
        assert b"Telemetry anomaly review" in response.read()
    print("Dashboard, nine models, all three detectors, and invalid-input responses verified.")


if __name__ == "__main__":
    check(sys.argv[1].rstrip("/") if len(sys.argv) > 1 else "http://127.0.0.1:8000")
