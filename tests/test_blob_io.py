import unittest
from unittest.mock import patch

import pandas as pd

from f1.helpers.blob_io import concat_blob_frames


class ConcatBlobFramesTests(unittest.TestCase):
    @patch("f1.blob_io.get_blob")
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


if __name__ == "__main__":
    unittest.main()
