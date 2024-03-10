import pandas as pd
from models.models import away_features, home_features, features
import src.dicts_and_lists as dal
import logging, coloredlogs

pd.set_option("display.max_rows", 1000)

# ------ Logger ------- #
logger = logging.getLogger("build_moving_average_model.py")
coloredlogs.install(level="INFO", logger=logger)

def build_moving_average_dataset(average_N, skip_n, leave_out=None):
    # FIXME fix using groupby rolling 
    df_2017 = pd.read_csv("src/past_data/2017-2018/split_stats_per_game.csv")
    df_2018 = pd.read_csv("src/past_data/2018-2019/split_stats_per_game.csv")
    df_2019 = pd.read_csv("src/past_data/2019-2020/split_stats_per_game.csv")
    df_2020 = pd.read_csv("src/past_data/2020-2021/split_stats_per_game.csv")
    df_2021 = pd.read_csv("src/past_data/2021-2022/split_stats_per_game.csv")

    logger.info(
        f"Averaging the datasets. MA: {average_N} games, first {skip_n} games are skipped."
    )
