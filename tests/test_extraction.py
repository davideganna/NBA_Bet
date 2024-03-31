"""

Unit tests regarding the Extraction module.

Run the tests rom the root folder using:
python -m unittest discover -s tests

"""

import unittest
from src.config_reader import config
from src.ETL.DataExtractor import Extraction


class TestExtraction(unittest.TestCase):
    """Unit tests regarding the Extraction module."""

    def __init__(self, methodName: str = "runTest") -> None:
        super().__init__(methodName)
        self.folder = f"src/past_data/{config['years']}/"
        self.season = config["season"]

    def test_get_current_month_data(self):
        extraction = Extraction(self.folder, self.season)
        df_month, current_month = extraction.get_current_month_data()
        cols = list(df_month.columns)

        # Ensure all columns are present
        self.assertEqual(
            cols,
            [
                "Date",
                "Start (ET)",
                "Visitor/Neutral",
                "PTS",
                "Home/Neutral",
                "PTS.1",
                "Unnamed: 6",
                "Unnamed: 7",
                "Attend.",
                "Arena",
                "Notes",
            ],
        )
