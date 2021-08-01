# ---------------------------- API.py ---------------------------- #
# Provides integration with the Basketball API found at:
# https://dashboard.api-football.com/basketball/
# ---------------------------------------------------------------- #
import http.client

def authenticate():
    url = 'v1.basketball.api-sports.io'
    with open('secrets/api_key', 'r') as f:
        key = f.readlines()[0]
        
    headers = {
        'x-rapidapi-host': url,
        'x-rapidapi-key': key 
    }
    
    conn = http.client.HTTPSConnection(url)
    return conn, headers

def get_next_games():
    pass

def get_odds():
    conn, headers = authenticate()
    conn.request("GET", "/odds?season=2019-2020&bet=1&bookmaker=6&game=1912&league=12", headers=headers)

    res = conn.getresponse()
    data = res.read()

    print(data.decode("utf-8"))


get_odds()
