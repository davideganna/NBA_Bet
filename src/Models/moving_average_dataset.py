import pandas as pd
from src.models.models import away_features, home_features, features
import src.dicts_and_lists as dal
import coloredlogs


def build_moving_average_dataset(config, logger, average_n: int):
    """
    The idea is to build a DataFrame that is team-agnostic, meaning that the team will not influence
        how the average dataset is created. When building such dataset, only the team features are
        considered in the calculation while the name of the team is left out.
        By training on this dataset, the algorithm learns that teams with these set of features will
        result in a given winner (home or away).

    On prediction phase, the average of the last N games for the team is calculated.
    """
    # ------ Logger ------- #
    coloredlogs.install(level="INFO", logger=logger)
    logger.info(f"Averaging the datasets. MA: {average_n} games.")

    dfs_list = []
    # TODO test with multiple years, currently testing just with one
    for year in config["moving_average_dataset"]:
        _df = pd.read_csv(f"src/past_data/{year}/split_stats_per_game.csv")
        _df["index"] = _df.index
        dfs_list.append(_df)

    dfs = pd.concat(dfs_list, ignore_index=True)

    away_rolling_df = (
        dfs.groupby(["season", "Team_away"])
        .rolling(average_n)[away_features]
        .agg("mean")
        .reset_index()
        .rename(columns={"level_2": "index_sspg"})
    )
    home_rolling_df = (
        dfs.groupby(["season", "Team_home"])
        .rolling(average_n)[home_features]
        .agg("mean")
        .reset_index()
        .rename(columns={"level_2": "index_sspg"})
    )

    average_df = pd.merge(
        away_rolling_df, home_rolling_df, on=["season", "index_sspg"]
    ).dropna()

    # Add target to average dataset
    average_df = pd.merge(
        average_df,
        dfs[["season", "index", "Winner"]],
        left_on=["season", "index_sspg"],
        right_on=["season", "index"],
    )

    file_name = f"average_{config['moving_average_dataset'][0][:4]}-{config['moving_average_dataset'][-1][-4:]}"
    average_df.to_csv(f"src/past_data/average_seasons/{file_name}.csv")
