import pandas as pd
import numpy as np
import logging, coloredlogs
from pandas.core.frame import DataFrame

pd.set_option("display.max_rows", 1000)

# ------ Logger ------- #
logger = logging.getLogger("elo_model.py")
coloredlogs.install(level="INFO", logger=logger)


def build_model(df: DataFrame):
    """
    Backtest on the DataFrame in input.
    """
    predictions = []
    for _n, _row in df.iterrows():
        # Predict new results based on Elo rating
        if _row["ModelProb_Away"] > _row["ModelProb_Home"]:
            predictions.append(1)
        else:
            predictions.append(0)

    df["Predictions"] = predictions

    ev_df = df.copy()

    # Hyperparameters
    margin = 0
    prob_limit = 0
    betting_limiter = True

    # Calculate accuracy of predicted teams, when they were the favorite by a margin
    correctly_predicted_amount = ev_df.loc[
        (ev_df["Predictions"] == ev_df["Winner"])
    ].count()

    wrongly_predicted_amount = ev_df.loc[
        (ev_df["Predictions"] != ev_df["Winner"])
    ].count()

    total_predicted = correctly_predicted_amount[0] + wrongly_predicted_amount[0]

    if correctly_predicted_amount[0] != 0 and total_predicted != 0:
        accuracy = correctly_predicted_amount[0] / total_predicted
        # logger.info(f'Accuracy when team is favorite, loser odds are greater than winner ones + margin ({margin}) and model probability is > {prob_limit}: {accuracy:.3f}')
    else:
        logger.info(
            "Accuracy could not be computed. You may try to relax the conditions (margin and/or prob_limit)."
        )

    correctly_pred_df = ev_df.loc[(ev_df["Predictions"] == ev_df["Winner"])]

    wrongly_pred_df = ev_df.loc[(ev_df["Predictions"] != ev_df["Winner"])]

    ev_df = (
        pd.concat([correctly_pred_df, wrongly_pred_df], axis=0)
        .sort_index()
        .reset_index(drop=True)
    )

    # Compare Predictions and TrueValues
    comparison_column = np.where(ev_df["Predictions"] == ev_df["Winner"], True, False)

    # Kelly's criterion: bet a different fraction of the bankroll depending on odds
    starting_bankroll = 100  # €
    current_bankroll = starting_bankroll
    bet_amount = []
    frac_bet = []  # Percentage of bankroll bet
    net_won = []
    bankroll = []

    for n, row in ev_df.iterrows():
        if row["Predictions"] == 0:
            frac_amount = (row["ModelProb_Home"] * row["OddsHome"] - 1) / (
                row["OddsHome"] - 1
            )
        elif row["Predictions"] == 1:
            frac_amount = (row["ModelProb_Away"] * row["OddsAway"] - 1) / (
                row["OddsAway"] - 1
            )

        if frac_amount > 0:
            # Limit the portion of bankroll to bet
            if (
                frac_amount > 0.2
                and current_bankroll < 2 * starting_bankroll
                and betting_limiter == True
            ):
                frac_amount = 0.2

            frac_bet.append(round(frac_amount, 2))

            # Max win is capped at 10000
            if (
                (current_bankroll * frac_amount * row["OddsHome"])
                and (row["Winner"] == 0)
            ) > 10000:
                bet_amount.append(int(10000 / row["OddsHome"]))
            elif (
                (current_bankroll * frac_amount * row["OddsAway"])
                and (row["Winner"] == 1)
            ) > 10000:
                bet_amount.append(int(10000 / row["OddsAway"]))
            # Min bet is 2€
            elif int(current_bankroll * frac_amount) >= 2:
                bet_amount.append(int(current_bankroll * frac_amount))
            elif int(current_bankroll * frac_amount) < 2:
                bet_amount.append(0)

            if row["Winner"] == 0:
                net_won.append(
                    bet_amount[n]
                    * row["OddsHome"]
                    * (row["Predictions"] == row["Winner"])
                    - bet_amount[n]
                )
            else:
                net_won.append(
                    bet_amount[n]
                    * row["OddsAway"]
                    * (row["Predictions"] == row["Winner"])
                    - bet_amount[n]
                )

            current_bankroll = current_bankroll + net_won[n]
            bankroll.append(current_bankroll)
        else:
            frac_bet.append(0)
            bet_amount.append(0)
            net_won.append(0)
            bankroll.append(current_bankroll)

    ev_df["FractionBet"] = frac_bet
    ev_df["BetAmount"] = bet_amount
    ev_df["NetWon"] = net_won
    ev_df["Bankroll"] = bankroll

    # Evaluate the bankroll and the ROI
    ev_df = ev_df[
        [
            "Date",
            "Team_away",
            "Team_home",
            "Predictions",
            "Winner",
            "OddsAway_Elo",
            "OddsHome_Elo",
            "ModelProb_Away",
            "ModelProb_Home",
            "OddsAway",
            "OddsHome",
            "FractionBet",
            "BetAmount",
            "NetWon",
            "Bankroll",
        ]
    ]

    return ev_df
