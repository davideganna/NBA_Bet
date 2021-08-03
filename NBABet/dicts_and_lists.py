# ------------------------------- dicts_and_lists.py ------------------------------- #
# Contains a collection of Lists and Dictionaries used throughout the program.
# ---------------------------------------------------------------------------------- #

##### Teams list #####
teams = [
    'Atlanta Hawks', 
    'Boston Celtics', 
    'Brooklyn Nets', 
    'Charlotte Hornets', 
    'Chicago Bulls', 
    'Cleveland Cavaliers', 
    'Dallas Mavericks', 
    'Denver Nuggets', 
    'Detroit Pistons', 
    'Golden State Warriors', 
    'Houston Rockets', 
    'Indiana Pacers', 
    'Los Angeles Clippers', 
    'Los Angeles Lakers', 
    'Memphis Grizzlies', 
    'Miami Heat', 
    'Milwaukee Bucks', 
    'Minnesota Timberwolves', 
    'New Orleans Pelicans', 
    'New York Knicks', 
    'Oklahoma City Thunder', 
    'Orlando Magic', 
    'Philadelphia 76ers', 
    'Phoenix Suns', 
    'Portland Trail Blazers', 
    'Sacramento Kings', 
    'San Antonio Spurs', 
    'Toronto Raptors', 
    'Utah Jazz', 
    'Washington Wizards'
]

##### Months Dictionary #####
months_dict = {
    'Jan' : '01',
    'Feb' : '02',
    'Mar' : '03',
    'Apr' : '04',
    'May' : '05',
    'Jun' : '06',
    'Jul' : '07',
    'Aug' : '08',
    'Sep' : '09',
    'Oct' : '10',
    'Nov' : '11',
    'Dec' : '12'
}

##### Teams Dictionary #####
teams_dict = {
    'Atlanta Hawks' : 'ATL', 
    'Boston Celtics' : 'BOS', 
    'Brooklyn Nets' : 'BRK', 
    'Charlotte Hornets' : 'CHO', 
    'Chicago Bulls' : 'CHI', 
    'Cleveland Cavaliers' : 'CLE', 
    'Dallas Mavericks' : 'DAL', 
    'Denver Nuggets' : 'DEN', 
    'Detroit Pistons' : 'DET', 
    'Golden State Warriors' : 'GSW', 
    'Houston Rockets' : 'HOU', 
    'Indiana Pacers' : 'IND', 
    'Los Angeles Clippers' : 'LAC', 
    'Los Angeles Lakers' : 'LAL', 
    'Memphis Grizzlies' : 'MEM', 
    'Miami Heat' : 'MIA', 
    'Milwaukee Bucks' : 'MIL', 
    'Minnesota Timberwolves' : 'MIN', 
    'New Orleans Pelicans' : 'NOP', 
    'New York Knicks' : 'NYK', 
    'Oklahoma City Thunder' : 'OKC', 
    'Orlando Magic' : 'ORL', 
    'Philadelphia 76ers' : 'PHI', 
    'Phoenix Suns' : 'PHO', 
    'Portland Trail Blazers' : 'POR', 
    'Sacramento Kings' : 'SAC', 
    'San Antonio Spurs' : 'SAS', 
    'Toronto Raptors' : 'TOR', 
    'Utah Jazz' : 'UTA', 
    'Washington Wizards' : 'WAS'
}

teams_to_int = {
    'Atlanta Hawks' : 0, 
    'Boston Celtics' : 1, 
    'Brooklyn Nets' : 2, 
    'Charlotte Hornets' : 3, 
    'Chicago Bulls' : 4, 
    'Cleveland Cavaliers' : 5, 
    'Dallas Mavericks' : 6, 
    'Denver Nuggets' : 7, 
    'Detroit Pistons' : 8, 
    'Golden State Warriors' : 9, 
    'Houston Rockets' : 10, 
    'Indiana Pacers' : 11, 
    'Los Angeles Clippers' : 12, 
    'Los Angeles Lakers' : 13, 
    'Memphis Grizzlies' : 14, 
    'Miami Heat' : 15, 
    'Milwaukee Bucks' : 16, 
    'Minnesota Timberwolves' : 17, 
    'New Orleans Pelicans' : 18, 
    'New York Knicks' : 19, 
    'Oklahoma City Thunder' : 20, 
    'Orlando Magic' : 21, 
    'Philadelphia 76ers' : 22, 
    'Phoenix Suns' : 23, 
    'Portland Trail Blazers' : 24, 
    'Sacramento Kings' : 25, 
    'San Antonio Spurs' : 26, 
    'Toronto Raptors' : 27, 
    'Utah Jazz' : 28, 
    'Washington Wizards' : 29
}

#### Data per games Dictionary ####
data_dict = {
    'Team' : [],
    'MP'   : [],
    'FG'   : [],
    'FGA'  : [],
    'FG%'  : [],
    '3P'   : [],
    '3PA'  : [],
    '3P%'  : [],
    'FT'   : [],
    'FTA'  : [],
    'FT%'  : [],
    'ORB'  : [],
    'DRB'  : [],
    'TRB'  : [],
    'AST'  : [],
    'STL'  : [],
    'BLK'  : [],
    'TOV'  : [],
    'PF'   : [],
    'PTS'  : [],
    '+/-'  : []
}

#### Columns of the data_dict (Used to create the stats_per_game DataFrame) ####
columns_data_dict = [
    'Team', 'MP', 'FG', 'FGA', 'FG%', '3P', '3PA', '3P%', 'FT', 'FTA', 'FT%', 'ORB', 'DRB', 'TRB', 'AST', 'STL', 'BLK', 'TOV', 'PF',  'PTS',  '+/-'
]

# Stats per Game - Away
spg_away = {
    'Team' : 'Team_away',
    'MP'   : 'MP_away',
    'FG'   : 'FG_away',
    'FGA'  : 'FGA_away',
    'FG%'  : 'FG%_away',
    '3P'   : '3P_away',
    '3PA'  : '3PA_away',
    '3P%'  : '3P%_away',
    'FT'   : 'FT_away',
    'FTA'  : 'FTA_away',
    'FT%'  : 'FT%_away',
    'ORB'  : 'ORB_away',
    'DRB'  : 'DRB_away',
    'TRB'  : 'TRB_away',
    'AST'  : 'AST_away',
    'STL'  : 'STL_away',
    'BLK'  : 'BLK_away',
    'TOV'  : 'TOV_away',
    'PF'   : 'PF_away',
    'PTS'  : 'PTS_away'
}

# Stats per Game - Home
spg_home = {
    'Team' : 'Team_home',
    'MP'   : 'MP_home',
    'FG'   : 'FG_home',
    'FGA'  : 'FGA_home',
    'FG%'  : 'FG%_home',
    '3P'   : '3P_home',
    '3PA'  : '3PA_home',
    '3P%'  : '3P%_home',
    'FT'   : 'FT_home',
    'FTA'  : 'FTA_home',
    'FT%'  : 'FT%_home',
    'ORB'  : 'ORB_home',
    'DRB'  : 'DRB_home',
    'TRB'  : 'TRB_home',
    'AST'  : 'AST_home',
    'STL'  : 'STL_home',
    'BLK'  : 'BLK_home',
    'TOV'  : 'TOV_home',
    'PF'   : 'PF_home',
    'PTS'  : 'PTS_home'
}

features_away = [
    'Team_away',
    'MP_away',
    'FG_away',
    'FGA_away',
    'FG%_away',
    '3P_away',
    '3PA_away',
    '3P%_away',
    'FT_away',
    'FTA_away',
    'FT%_away',
    'ORB_away',
    'DRB_away',
    'TRB_away',
    'AST_away',
    'STL_away',
    'BLK_away',
    'TOV_away',
    'PF_away'
    ]

features_home = [
    'Team_home',
    'MP_home',
    'FG_home',
    'FGA_home',
    'FG%_home',
    '3P_home',
    '3PA_home',
    '3P%_home',
    'FT_home',
    'FTA_home',
    'FT%_home',
    'ORB_home',
    'DRB_home',
    'TRB_home',
    'AST_home',
    'STL_home',
    'BLK_home',
    'TOV_home',
    'PF_home'
    ]

# Historical Odds
historical_away = {
    'Team'  : 'Team_away',
    'Final' : 'Final_away',
    'Odds'  : 'Odds_away'
}

historical_home = {
    'Team'  : 'Team_home',
    'Final' : 'Final_home',
    'Odds'  : 'Odds_home'
}