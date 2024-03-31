"""Unit tests regarding the Transformation module."""

import unittest
from src.config_reader import config
from src.ETL.DataTransformer import Transformation

import pandas as pd

class TestTransformation(unittest.TestCase):
    """Unit tests regarding the Transformation module."""

    def __init__(self, methodName: str = "runTest") -> None:
        super().__init__(methodName)
        self.folder = f"src/past_data/{config['years']}/"

    def test_polish_df_month(self):
        transformation = Transformation(self.folder)

        df_test = pd.DataFrame(
            {
                "Visitor/Neutral": ['A', 'B', 'C'],
                "Home/Neutral": ['B', 'C', 'A'],
                "PTS": [80, 90, None],
                "PTS.1": [90, None, 100],
            }
        )

        df_polished, csv_path = transformation.polish_df_month(df_test, current_month='month')

        # Assert no NaNs
        n_nans = df_polished.dropna().shape[0]
        print(n_nans)
        self.assertTrue(n_nans, 0)