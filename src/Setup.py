# ------------------------------- setup.py ------------------------------- #
# To be run only once, before main.py is run.
# It creates the required .csv datasets which will be accessed in main.py
# ------------------------------------------------------------------------ #
import helper as helper
import logging, coloredlogs
import yaml

# ------ Logger ------- #
logger = logging.getLogger("setup.py")
coloredlogs.install(level="DEBUG")

with open("src/configs/main_conf.yaml") as f:
    config = yaml.safe_load(f)

years = config["years"]

folder = (
    f"past_data/{years}/"  # Specify the current NBA season to save the .csv datasets.
)
# Elo setup
helper.build_elo_csv(years)

# To build the stats_per_game.csv
logger.warning(
    "\nDo you want to build stats_per_game.csv? [y/n]\n \
               WARNING: In order to respect the server load, requests are sent ever 10s: this process will take a lot of time."
)
inp = input()
if inp == "y":
    # FIXME this has been substituted by src/get_past_datasets.py
    helper.build_stats_per_game_csv(folder)
