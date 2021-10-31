from datetime import datetime, timedelta
import requests

class Api:
    """
    Base class for interfacing with the Basketball API.
    """
    def __init__(self):
        self.url = 'https://v1.basketball.api-sports.io/'
        with open('secrets/api_key') as f:
            self.api_key = f.readline()
        self.league = '12' # NBA League
        self.season = '2021-2022'
        self.headers = {
            'x-rapidapi-key'  : self.api_key,
            'x-rapidapi-host' : self.url
        }


    def get_tomorrows_games(self):
        date = 'date=' + (datetime.today() + timedelta(days=1)).strftime('%Y-%m-%d')
        endpoint = 'games?' + date + '&league=' + self.league + '&season=' + self.season
        query = self.url + endpoint
        payload = {}
        response = requests.request("GET", query, headers=self.headers, data=payload).json()
        
        # Next games organized as a dictionary with keys = HomeTeam --> values: AwayTeam
        next_games = {}
        for match in response['response']:
            home_team = match['teams']['home']['name']
            away_team = match['teams']['away']['name']
            next_games[home_team] = away_team
        return next_games


        
