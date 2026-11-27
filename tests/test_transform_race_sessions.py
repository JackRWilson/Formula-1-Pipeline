import unittest
from unittest.mock import patch

import pandas as pd

from f1.transform import TransformQualResult, TransformRaceResult


class TransformRaceSessionTests(unittest.TestCase):
    def test_race_and_sprint_grid_positions_join_by_session_type(self):
        results = pd.DataFrame(
            [
                {
                    "race_id": "Singapore_2026",
                    "year": 2026,
                    "round": 17,
                    "driver_name": "Max Verstappen",
                    "driver_code": "VER",
                    "team_name": "Red Bull Racing",
                    "end_position": 1,
                    "points": 25,
                    "laps_completed": 62,
                    "session_type": "race",
                },
                {
                    "race_id": "Singapore_2026",
                    "year": 2026,
                    "round": 17,
                    "driver_name": "Max Verstappen",
                    "driver_code": "VER",
                    "team_name": "Red Bull Racing",
                    "end_position": 1,
                    "points": 8,
                    "laps_completed": 20,
                    "session_type": "sprint",
                },
            ]
        )
        grids = pd.DataFrame(
            [
                {
                    "race_id": "Singapore_2026",
                    "round": 17,
                    "driver_name": "Max Verstappen",
                    "driver_code": "VER",
                    "team_name": "Red Bull Racing",
                    "start_position": 2,
                    "session_type": "race",
                },
                {
                    "race_id": "Singapore_2026",
                    "round": 17,
                    "driver_name": "Max Verstappen",
                    "driver_code": "VER",
                    "team_name": "Red Bull Racing",
                    "start_position": 1,
                    "session_type": "sprint",
                },
            ]
        )
        transformer = TransformRaceResult()

        results, grids = transformer._consolidate_columns(results, grids)
        joined = transformer._join_starting_grid(results, grids)

        self.assertEqual(
            joined.set_index("session_type")["starting_position"].to_dict(),
            {"race": 2, "sprint": 1},
        )

    def test_fastf1_sprint_fastest_lap_uses_fastest_valid_lap_per_driver(self):
        laps = pd.DataFrame(
            {
                "Driver": ["VER", "VER", "VER", "VER", "HAM", "HAM"],
                "LapTime": pd.to_timedelta(
                    [90.2, 89.5, 88.0, 87.0, 91.0, 85.0], unit="s"
                ),
                "LapNumber": [1, 2, 3, 4, 1, 2],
                "TrackStatus": ["1", "1", "1", "2", "1", "1"],
                "PitOutTime": [pd.NaT, pd.NaT, pd.NaT, pd.NaT, pd.NaT, pd.NaT],
                "PitInTime": [pd.NaT, pd.NaT, pd.NaT, pd.NaT, pd.NaT, pd.NaT],
                "IsAccurate": [True, True, False, True, True, True],
                "season_year": [2026] * 6,
                "round_number": [17] * 6,
                "session_name": ["Sprint"] * 6,
            }
        )
        results = pd.DataFrame(
            {
                "race_id": ["Singapore_2026", "Singapore_2026"],
                "season_year": [2026, 2026],
                "round": [17, 17],
            }
        )

        with patch("f1.transform.load_blobs", return_value=laps):
            fastest = TransformRaceResult()._extract_fastf1_sprint_fastest_laps(
                results
            )

        actual = fastest.set_index("driver_code")[
            ["fastest_lap_seconds", "fastest_lap_number", "session_type", "race_id"]
        ].to_dict("index")
        self.assertEqual(actual["VER"]["fastest_lap_seconds"], 89.5)
        self.assertEqual(actual["VER"]["fastest_lap_number"], 2)
        self.assertEqual(actual["VER"]["session_type"], "sprint")
        self.assertEqual(actual["VER"]["race_id"], "Singapore_2026")
        self.assertEqual(actual["HAM"]["fastest_lap_seconds"], 85.0)

    def test_qualifying_transform_loads_both_qualifying_session_folders(self):
        qualifying = pd.DataFrame(
            {
                "race_id": ["Singapore_2026", "Singapore_2026"],
                "driver_name": ["Max Verstappen", "Max Verstappen"],
                "team_name": ["Red Bull Racing", "Red Bull Racing"],
                "q1_time": ["1:33.477", "1:33.477"],
                "q2_time": ["1:32.701", "1:32.701"],
                "q3_time": ["1:31.156", "1:31.156"],
                "qual_position": [1, 1],
                "qual_laps": [15, 15],
                "driver_code": ["VER", "VER"],
                "session_type": ["qualifying", "sprint-qualifying"],
            }
        )

        with patch("f1.transform.load_blobs", return_value=qualifying) as load:
            result = TransformQualResult()._extract_qual_results()

        self.assertEqual(
            result["session_type"].tolist(), ["qualifying", "sprint_qualifying"]
        )
        self.assertEqual(
            load.call_args.kwargs["folders"],
            ("qualifying", "sprint-qualifying"),
        )


if __name__ == "__main__":
    unittest.main()
