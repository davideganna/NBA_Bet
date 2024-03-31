from src.ETL.DataExtractor import Extraction
from src.ETL.DataTransformer import Transformation
from src.ETL.DataLoader import Loading
from src.config_reader import config

from src.models.moving_average_dataset import build_moving_average_dataset

# --------------- ETL Pipeline --------------- #
def etl_pipeline(folder, logger):
    extraction = Extraction(folder, config["season"])
    transformation = Transformation(folder)
    loading = Loading(folder)

    df_month, current_month = extraction.get_current_month_data()
    df_month, csv_path = transformation.polish_df_month(df_month, current_month)
    loading.save_df_month(df_month, current_month, csv_path)

    # Update the training dataset
    avg_df = build_moving_average_dataset(logger, 2)

    return avg_df
