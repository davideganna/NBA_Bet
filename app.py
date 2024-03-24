#    888b    888 888888b.          d8888     888888b.            888
#    8888b   888 888  "88b        d88888     888  "88b           888
#    88888b  888 888  .88P       d88P888     888  .88P           888
#    888Y88b 888 8888888K.      d88P 888     8888888K.   .d88b.  888888
#    888 Y88b888 888  "Y88b    d88P  888     888  "Y88b d8P  Y8b 888
#    888  Y88888 888    888   d88P   888     888    888 88888888 888
#    888   Y8888 888   d88P  d8888888888     888   d88P Y8b.     Y88b.
#    888    Y888 8888888P"  d88P     888     8888888P"   "Y8888   "Y888


import logging, coloredlogs
import pandas as pd
import yaml
from src.api import Api
from src.models.moving_average_dataset import build_moving_average_dataset
from src.telegram import telegramBot
from src.ETL.DataExtractor import Extraction
from src.ETL.DataTransformer import Transformation
from src.ETL.DataLoader import Loading
from src.config_reader import config
from src.build_predictions import predict_on_elo


# TODO move to another folder
# --------------- ETL Pipeline --------------- #
def etl_pipeline(logger):
    df_month, current_month = Extraction.get_current_month_data()
    df_month, csv_path = Transformation.polish_df_month(df_month, current_month)
    Loading.save_df_month(df_month, current_month, csv_path)
    # Update the training dataset
    avg_df = build_moving_average_dataset(logger, 2)
    return avg_df


# --------------- Logger --------------- #
logger = logging.getLogger(__name__)
coloredlogs.install(level="INFO")

folder = (
    f"src/past_data/{config['years']}/"
)

try:
    path = f"{folder}{config['years']}_season.csv"
    season_df = pd.read_csv(path)
except Exception as exc:
    logger.error(
        'The program could not access the .csv datasets. Be sure to run "setup.py" before "main.py".\n'
        "Alternatively, check that the path to the folder where the .csv files are located is correct.\n"
        f'Exception: "{exc}"\n'
        f"Path: {path}"
    )
else:
    Extraction = Extraction(folder, config["years"][:4])
    Transformation = Transformation(folder)
    Loading = Loading(folder)

    # Full ETL Pipeline
    avg_df = etl_pipeline(logger)

    # Get tonight's games
    next_games = Api().get_tonights_games()
    
    # Predict winner
    team_to_prob = predict_on_elo(avg_df, next_games)

    telegramBot().send_message(next_games, team_to_prob)
