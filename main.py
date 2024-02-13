#    888b    888 888888b.          d8888     888888b.            888   
#    8888b   888 888  "88b        d88888     888  "88b           888   
#    88888b  888 888  .88P       d88P888     888  .88P           888   
#    888Y88b 888 8888888K.      d88P 888     8888888K.   .d88b.  888888
#    888 Y88b888 888  "Y88b    d88P  888     888  "Y88b d8P  Y8b 888   
#    888  Y88888 888    888   d88P   888     888    888 88888888 888   
#    888   Y8888 888   d88P  d8888888888     888   d88P Y8b.     Y88b. 
#    888    Y888 8888888P"  d88P     888     8888888P"   "Y8888   "Y888


from apscheduler.schedulers.background import BackgroundScheduler
import logging, coloredlogs
import pandas as pd
import os
from src.Api import Api
from src.Models.moving_average_dataset import build_moving_average_dataset
from src.Telegram import TelegramBot
from src.ETL import DataExtractor, DataTransformer, DataLoader


# --------------- ETL Pipeline --------------- #
def etl_pipeline():
    df_month, current_month = Extraction.get_current_month_data()
    df_month, csv_path = Transformation.polish_df_month(df_month, current_month)
    Loading.save_df_month(df_month, current_month, csv_path)
    # Update the training dataset
    build_moving_average_dataset(3, 0)


# --------------- Scheduler --------------- #
sched = BackgroundScheduler(daemon=True)
sched.add_job(lambda: etl_pipeline(), 'cron', hour=12, minute=00)
sched.add_job(lambda: TelegramBot().send_message(Api().get_tonights_games()), 'cron', hour=12, minute=30)


# --------------- Logger --------------- #
logger = logging.getLogger(__name__)
coloredlogs.install(level='INFO')


# --------------- Main --------------- #
years = os.environ.get('years')
season = os.environ.get('season')
folder = 'src/past_data/' + years + '/' # Specify the current NBA season to save the .csv datasets.

Extraction = DataExtractor.Extraction(folder, season)
Transformation = DataTransformer.Transformation(folder)
Loading = DataLoader.Loading(folder, years)

try:
    season_df = pd.read_csv(folder + years + '_season.csv')
except Exception as exc:
    logger.error(
        'The program could not access the .csv datasets. Be sure to run "Setup.py" before "main.py".\n'\
        'Alternatively, check that the path to the folder where the .csv files are located is correct.\n' \
        f'Exception: "{exc}"'
    )
else:
    # Full ETL Pipeline
    etl_pipeline()
    
    # Telegram integration
    TelegramBot().send_message(Api().get_tonights_games())

    # ----- If you don't want to run the program at fixed times, comment the lines below. ----- #
    sched.start()
    logger.warning('Type "exit" to stop the program.\n')
    inp = input()
    while inp != 'exit':
        logger.warning('Program is running. Type "exit" to stop the program.\n')
        inp = input()