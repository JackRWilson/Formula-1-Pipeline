import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pandas as pd

from f1.helpers.blob_io import concat_blob_frames, load_blobs
from f1.helpers.constants import RACE_RESULT_FOLDERS, SESSION_SOURCE


class ConcatBlobFramesTests(unittest.TestCase):
    @patch("f1.helpers.blob_io.get_blob")
    def test_concatenates_every_blob_and_adds_round(self, get_blob):
        blob_index = pd.DataFrame(
            {
                "blob_name": ["race/2018/1.parquet", "race/2018/2.parquet"],
                "round": [1, 2],
            }
        )
        get_blob.side_effect = [
            pd.DataFrame({"race_id": ["2018-1"]}),
            pd.DataFrame({"race_id": ["2018-2"]}),
        ]

        result = concat_blob_frames(blob_index, {"race_id"}, add_round=True)

        self.assertEqual(result["race_id"].tolist(), ["2018-1", "2018-2"])
        self.assertEqual(result["round"].tolist(), [1, 2])
        self.assertEqual(get_blob.call_count, 2)

    @patch("f1.helpers.blob_io.get_blob")
    def test_load_blobs_lists_filters_and_transforms(self, get_blob):
        container = Mock()
        container.list_blobs.return_value = [
            SimpleNamespace(name="race-result/2020/01.parquet"),
            SimpleNamespace(name="qualifying/2020/01.parquet"),
        ]
        get_blob.return_value = pd.DataFrame(
            {"driver_name": [" Max Verstappen "], "driver_code": ["VER"]}
        )

        result = load_blobs(
            SESSION_SOURCE,
            {"driver_name"},
            folders=RACE_RESULT_FOLDERS,
            filters={"session": "race-result"},
            transform=lambda data: data[["driver_name"]],
            container_client=container,
        )

        container.list_blobs.assert_called_once_with(
            name_starts_with="race-result/"
        )
        self.assertEqual(result["driver_name"].tolist(), [" Max Verstappen "])
        get_blob.assert_called_once_with("race-result/2020/01.parquet")


if __name__ == "__main__":
    unittest.main()
