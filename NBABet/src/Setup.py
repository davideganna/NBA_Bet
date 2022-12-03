# ------------------------------- Setup.py ------------------------------- #
# To be run only once, before main.py is run.                              
# It creates the required .csv datasets which will be accessed in main.py  
# ------------------------------------------------------------------------ #
import src.Helper as Helper
import logging, coloredlogs

# ------ Logger ------- #
logger = logging.getLogger('Setup.py')
coloredlogs.install(level='DEBUG')

folder = 'src/past_data/2021-2022/' # Specify the current NBA season to save the .csv datasets.
# Elo Setup
Helper.build_elo_csv()

# To build the stats_per_game.csv
logger.warning('\nDo you want to build stats_per_game.csv? [y/n]\nWARNING: This process takes a lot of time.')
inp = input()
if inp == 'y':
    Helper.build_stats_per_game_csv(folder)