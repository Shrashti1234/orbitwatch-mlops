import unittest

import numpy as np
import pandas as pd

from orbitwatch.detector import WINDOW, alert_ranges, evaluate, make_features


class DetectorTests(unittest.TestCase):
    def test_later_readings_cannot_change_features_for_earlier_readings(self):
        readings = np.arange(2 * WINDOW, dtype=float)
        changed = readings.copy()
        changed[-1] *= 8
        pd.testing.assert_frame_equal(make_features(readings).iloc[:-1], make_features(changed).iloc[:-1])
        self.assertEqual(make_features(readings).index[0], WINDOW)

    def test_alert_ranges_are_inclusive_and_skip_short_runs(self):
        flags = np.array([True, True, True, False, True, False, True, True, True])
        self.assertEqual(alert_ranges(flags), [(0, 2), (6, 8)])

    def test_event_detection_counts_extra_alerts_separately(self):
        scores = np.array([np.nan, 5, 5, 5, 0, 5, 5, 5, 0], dtype=float)
        result, events = evaluate(scores, 4, [(3, 4)])
        self.assertEqual(events, [(1, 3), (5, 7)])
        self.assertEqual(result["events_detected"], 1)
        self.assertEqual(result["unmatched_alert_intervals"], 1)
