import unittest
from unittest.mock import patch

import pandas as pd

from f1.transform import (
    TransformDriverSessionTirePerformance,
    TransformPracticeResult,
    TransformRaceControlSummary,
    TransformWeatherSummary,
    _normalize_session_types,
)


class TransformSessionNameTests(unittest.TestCase):
    def test_normalizes_source_session_aliases(self):
        sessions = pd.Series(
            [
                "FP1",
                "Practice 2",
                "practice/0",
                "Qualifying",
                "Sprint Qualifying",
                "Sprint Shootout",
                "Sprint",
                "Race",
            ]
        )

        self.assertEqual(
            _normalize_session_types(sessions).tolist(),
            [
                "practice_1",
                "practice_2",
                "practice_3",
                "qualifying",
                "sprint_qualifying",
                "sprint_qualifying",
                "sprint",
                "race",
            ],
        )

    def test_rejects_unrecognized_session_type(self):
        with self.assertRaisesRegex(ValueError, "Unrecognized session types"):
            _normalize_session_types(pd.Series(["unknown-session"]))

    def test_practice_results_use_canonical_session_type(self):
        practice = pd.DataFrame(
            {
                "race_id": ["Singapore_2026"],
                "session_type": ["FP3"],
                "lap_time": ["1:30.000"],
                "round": [17],
            }
        )

        result = TransformPracticeResult._best_practice_laps(practice)

        self.assertEqual(result["session_type"].tolist(), ["practice_3"])

    def test_weather_sessions_use_canonical_session_type(self):
        weather = pd.DataFrame(
            {
                "AirTemp": [30.0, 29.0],
                "Humidity": [50.0, 55.0],
                "Pressure": [1000.0, 1001.0],
                "Rainfall": [False, False],
                "TrackTemp": [40.0, 39.0],
                "WindSpeed": [1.0, 1.2],
                "season_year": [2026, 2026],
                "round_number": [17, 17],
                "session_name": ["Practice 1", "Sprint Shootout"],
            }
        )

        with patch("f1.transform.load_blobs", return_value=weather):
            result = TransformWeatherSummary()._extract_weather()

        self.assertEqual(
            result["session_type"].tolist(), ["practice_1", "sprint_qualifying"]
        )

    def test_race_control_messages_normalize_before_grouping(self):
        messages = pd.DataFrame(
            {
                "Time": [pd.Timedelta(seconds=1)],
                "Category": ["Flag"],
                "Message": ["Yellow"],
                "Status": ["ACTIVE"],
                "Flag": ["YELLOW"],
                "Lap": [1],
                "season_year": [2026],
                "round_number": [17],
                "session_name": ["Sprint Shootout"],
            }
        )

        with patch("f1.transform.load_blobs", return_value=messages):
            result = TransformRaceControlSummary()._extract_messages()

        self.assertEqual(result["session_name"].tolist(), ["sprint_qualifying"])

    def test_tire_laps_use_canonical_session_name(self):
        laps = pd.DataFrame(
            {
                "Driver": ["VER"],
                "LapTime": [pd.Timedelta(seconds=90)],
                "LapNumber": [1],
                "Compound": ["SOFT"],
                "TrackStatus": ["1"],
                "PitOutTime": [pd.NaT],
                "PitInTime": [pd.NaT],
                "IsAccurate": [True],
                "season_year": [2026],
                "round_number": [17],
                "session_name": ["Sprint Shootout"],
            }
        )

        with patch("f1.transform.load_blobs", return_value=laps):
            result = TransformDriverSessionTirePerformance()._extract_laps()

        self.assertEqual(result["session_name"].tolist(), ["sprint_qualifying"])


if __name__ == "__main__":
    unittest.main()
