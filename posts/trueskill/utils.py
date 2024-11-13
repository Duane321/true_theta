import requests
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
import json
from tqdm import tqdm
from sklearn.model_selection import train_test_split

def extract_boxing_record(url):
    """
    Given a boxer's Wikipedia URL, this will extract the table called "Professional boxing record" and do some cleanup.
    url: a wiki url for a fighter
    """
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    
    section = None
    for header in soup.find_all(['h2', 'h3', 'h4']):
        if 'Professional boxing record' in header.get_text():
            section = header
            break
    
    if section:
        tables = section.find_all_next('table')
        
        for table in tables:
            first_row = table.find('tr')
            columns = first_row.find_all(['th', 'td'])
            
            if len(columns) >= 4:
                headers = [header.get_text(strip=True) for header in table.find_all('th')]
                rows = []
                for row in table.find_all('tr')[1:]:  # Skip header row if present
                    cells = row.find_all(['th', 'td'])
                    rows.append([cell.get_text(strip=True) for cell in cells])
                df = pd.DataFrame(rows, columns=headers if headers else None).rename(columns={'Date': "Date raw", 'Res.':'Result'})
                return df 
                
    print('No suitable table found with at least 4 columns.')
    return None

def parse_dates(date_list):
    """
    Parse a list of dates and convert to datetime type
    date_list: a list of game dates
    """
    date_series = pd.Series(date_list)
    parsed_dates = pd.to_datetime(date_series.str.replace('–', '-').str.replace('[377]', ''), errors='coerce')
    return parsed_dates


def extract_ufc_record(url):
    """
    Given a ufc's Wikipedia URL, this will extract the table called "Mixed martial arts record" and do some cleanup.
    url: a wiki url for a fighter
    """
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    
    section = None
    for header in soup.find_all(['h2', 'h3', 'h4']):
        if 'Mixed martial arts record' in header.get_text():
            section = header
            break
    
    if section:
        tables = section.find_all_next('table')
        
        for table in tables:
            first_row = table.find('tr')
            columns = first_row.find_all(['th', 'td'])
            
            if len(columns) >= 4:
                headers = [header.get_text(strip=True) for header in table.find_all('th')]
                rows = []
                for row in table.find_all('tr')[1:]:  # Skip header row if present
                    cells = row.find_all(['th', 'td'])
                    rows.append([cell.get_text(strip=True) for cell in cells])
                df = pd.DataFrame(rows, columns=headers if headers else None).rename(columns={'Date': "Date raw", 'Res.':'Result'})
                return df 
                
    print('No suitable table found with at least 4 columns.')
    return None

def prepare_warcraft3_data(raw_data):
    """
    Prepare raw data to be split by train/test set used in trueskill through time algo
    raw_data: raw input data from warcraft3 game history
    """
    game_composition = []
    times = []
    np.random.seed(0)

    for _, row in raw_data.iterrows():
        c1, c2, c1s, c2s, t = row['competitor_1'], row['competitor_2'], row['competitor_1_score'], row['competitor_2_score'], row['timestamp']

        assert c1s == int(c1s)
        assert c2s == int(c2s)
        
        comp = [(c1, c2)] * int(c1s) + [(c2, c1)] * int(c2s)
        comp = np.random.permutation(comp).tolist() # Game order matters and we don't actually know it, so we randomize over it.

        for cp_g in comp:
            game_composition.append(cp_g)
            times.append(t)

    games = pd.DataFrame(game_composition, columns=['winner', 'loser']).assign(timestamp = times)

    return games

def prepare_tennis_data(games_filtered, target_players_lst):
    """
    Prepare raw data to be split by train/test set used in trueskill through time algo
    It should be called by TennisSpliter.train_test_split from train_test_split_game.py
    games: filtered input data from tennis game history with a target players list
    """

    # Create a dataframe for winners
    winners = games_filtered[['winner', 'timestamp', 'match_id']].copy()
    winners['result'] = 1
    winners = winners.rename(columns={'winner': 'player'})

    # Create a dataframe for losers
    losers = games_filtered[['loser', 'timestamp', 'match_id']].copy()
    losers['result'] = 0
    losers = losers.rename(columns={'loser': 'player'})

    # Concatenate winners and losers dataframes
    result_df = pd.concat([winners, losers], ignore_index=True)

    # Sort the resulting dataframe by timestamp
    result_df = result_df.sort_values(['player', 'timestamp']).reset_index(drop=True)

    players_ge_40_matches_df = result_df[result_df.player.isin(target_players_lst)].reset_index().iloc[:, 1:]

    return players_ge_40_matches_df

def prepare_boxing_data(games_filtered, target_players_lst):
    """
    Prepare raw data to be split by train/test set used in trueskill through time algo
    It should be called by BoxingSpliter.train_test_split from train_test_split_game.py
    games: filtered input data from tennis game history with a target players list
    """
    winners = games_filtered[['winner', 'timestamp', 'game_index']].copy()
    winners['result'] = 1
    winners = winners.rename(columns={'winner': 'player'})

    # Create a dataframe for losers
    losers = games_filtered[['loser', 'timestamp', 'game_index']].copy()
    losers['result'] = 0
    losers = losers.rename(columns={'loser': 'player'})

    # Concatenate winners and losers dataframes
    result_df = pd.concat([winners, losers], ignore_index=True)

    # Sort the resulting dataframe by timestamp
    result_df = result_df.sort_values(['player', 'timestamp']).reset_index(drop=True)

    players_ge_40_matches_df = result_df[result_df.player.isin(target_players_lst)].reset_index().iloc[:, 1:]

    return players_ge_40_matches_df




def train_test_split_by_players(players_lst, filtered_matches_df, test_size=0.2):
    """
    do a train_test_split by a list of players
    for each player, use the previous (1-test_size) as the train set and the last test_size as the test set
    """

    train_data = []
    test_data = []

    # for each player, we split the player's game history by timestamp
    for player in tqdm(players_lst):
        player_data = filtered_matches_df[filtered_matches_df['player'] == player].sort_values('timestamp')
        
        # Ensure we have enough data to split
        if len(player_data) > 1:
            player_train, player_test = train_test_split(player_data, test_size=test_size, shuffle=False)
            train_data.append(player_train)
            test_data.append(player_test)
        else:
            print(f'player %s has only one match'%(player))
            # If only one match, add it to training data
            train_data.append(player_data)

    train_df = pd.concat(train_data, ignore_index=True)
    test_df = pd.concat(test_data, ignore_index=True)

    return train_df, test_df
