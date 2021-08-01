# ------------------------------- Setup.py ------------------------------- #
# To be run only once, before main.py is run.                              
# It creates the required .csv datasets which will be accessed in main.py  
# ------------------------------------------------------------------------ #
import Helper
import logging, coloredlogs

# ------ Logger ------- #
logger = logging.getLogger('Setup.py')
coloredlogs.install(level='DEBUG')

# Elo Setup
elo_df = Helper.elo_setup()

folder = 'past_data/2020_2021/' # Specify the current NBA season to save the .csv datasets.
# DataFrame Setup
season_df = Helper.build_season_df(folder)
Helper.update_elo_csv(season_df)

# To build the stats_per_game.csv
logger.warning('\nDo you want to build stats_per_game.csv? [y/n]\nWARNING: This process takes a lot of time.')
inp = input()
if inp == 'y':
    Helper.build_stats_per_game_csv(folder)