from abc import ABC, abstractmethod
from utils import *
import json



class DataSpliter(ABC):
    def __init__(self, input_filename: str):
        self.input_filename = input_filename

    def prepare_train_set(self, input_data=None) -> pd.DataFrame:
        print("optional implementation for prepare_train_set")

    def prepare_test_set(self, input_data=None) -> pd.DataFrame:
        print("optional implementation for prepare_test_set")

    def train_test_split(self, input_data=None) -> pd.DataFrame:
        print("optional implementation for train_test_split")

class Warcraft3Spliter(DataSpliter):
    """
    assume the warcraft3 data is sorted in chronical order
    """
    def prepare_train_set(self, start_idx, end_idx, input_data=None) -> pd.DataFrame:
        """
        start_idx - starting index of train set, it is counted in reverse order
        end_idx - starting index of test set, it is counted in reverse order(the end_idx is exclusive)
        """
        #e.g. 'data/warcraft3.csv'
        games_raw = pd.read_csv(self.input_filename).query('(competitor_1_score > -0.0001) & (competitor_2_score > -0.0001)').iloc[start_idx:end_idx]
        games_raw['timestamp'] = pd.to_datetime(games_raw['date'])

        games_train = prepare_warcraft3_data(games_raw)

        return games_train

    def prepare_test_set(self, start_idx, end_idx=None, input_data=None) -> pd.DataFrame:
        """
        start_idx - starting index of train set, it is counted in reverse order
        end_idx - starting index of test set, it is counted in reverse order(the end_idx is exclusive)
        """
        #e.g. 'data/warcraft3.csv'
        if not end_idx:
            games_raw = pd.read_csv(self.input_filename).query('(competitor_1_score > -0.0001) & (competitor_2_score > -0.0001)').iloc[start_idx:]
        else:
            games_raw = pd.read_csv(self.input_filename).query('(competitor_1_score > -0.0001) & (competitor_2_score > -0.0001)').iloc[start_idx:end_idx]
        games_raw['timestamp'] = pd.to_datetime(games_raw['date'])

        games_test = prepare_warcraft3_data(games_raw)

        return games_test
    

class TennisSpliter(DataSpliter):
    def __init__(self, input_filename: str, target_players_filename: str):
        super().__init__(input_filename)
        self.target_players_filename = target_players_filename
        
    def train_test_split(self, test_size=0.2):
        #e.g. "data/tennis_players_ge_40_matches_lst.json"
        with open(self.target_players_filename, "r") as f:
            players_lst = json.load(f)
        #e.g. "data/tennis_matches_refined_tstt.parquet"
        games = pd.read_parquet(self.input_filename)
        games_filtered = games[games.winner.isin(players_lst) | games.loser.isin(players_lst)]

        players_filtered_matches_df = prepare_tennis_data(games_filtered, players_lst)

        train_df, test_df = train_test_split_by_players(players_filtered_matches_df['player'].unique(), players_filtered_matches_df, test_size)

        #merge the entire filtered games data with the selected train data
        games_filtered_train_df_w = pd.merge(games_filtered[['winner', 'loser', 'match_id', 'timestamp']], train_df[['player', 'match_id', 'timestamp']], left_on=['winner', 'match_id', 'timestamp'], right_on=['player', 'match_id', 'timestamp'], how='inner')
        games_filtered_train_df_l = pd.merge(games_filtered[['winner', 'loser', 'match_id', 'timestamp']], train_df[['player', 'match_id', 'timestamp']], left_on=['loser', 'match_id', 'timestamp'], right_on=['player', 'match_id', 'timestamp'], how='inner')
        games_filtered_train_df = pd.concat([games_filtered_train_df_w[['winner', 'loser', 'match_id', 'timestamp']], games_filtered_train_df_l[['winner', 'loser', 'match_id', 'timestamp']]]).drop_duplicates().sort_values('timestamp')

        #merge the entire filtered games data with the selected test data
        games_filtered_test_df_w = pd.merge(games_filtered[['winner', 'loser', 'match_id', 'timestamp']], test_df[['player', 'match_id', 'timestamp']], left_on=['winner', 'match_id', 'timestamp'], right_on=['player', 'match_id', 'timestamp'], how='inner')
        games_filtered_test_df_l = pd.merge(games_filtered[['winner', 'loser', 'match_id', 'timestamp']], test_df[['player', 'match_id', 'timestamp']], left_on=['loser', 'match_id', 'timestamp'], right_on=['player', 'match_id', 'timestamp'], how='inner')
        games_filtered_test_df = pd.concat([games_filtered_test_df_w[['winner', 'loser', 'match_id', 'timestamp']], games_filtered_test_df_l[['winner', 'loser', 'match_id', 'timestamp']]]).drop_duplicates().sort_values('timestamp')

        # filter out games appearing in both test and train set
        # it could happen that player A's last X% of games in the test set occur in player B's first 1-X% games in the train set
        games_filtered_test_unique_df = games_filtered_test_df[~games_filtered_test_df.match_id.isin(games_filtered_train_df.match_id.tolist())] 

        return games_filtered_train_df, games_filtered_test_unique_df
    

class BoxingSpliter(DataSpliter):
    def __init__(self, input_filename: str, target_players_filename: str):
        super().__init__(input_filename)
        self.target_players_filename = target_players_filename
        
    def train_test_split(self, test_size=0.2):
        #e.g. "data/players_ge_40_matches_lst.json"
        with open(self.target_players_filename, "r") as f:
            players_lst = json.load(f)
        #e.g. "data/boxing_matches_refined_tstt.parquet"
        games = pd.read_parquet(self.input_filename)

        games_filtered = games[games.winner.isin(players_lst) | games.loser.isin(players_lst)]
        players_filtered_matches_df = prepare_tennis_data(games_filtered, players_lst)

        train_df, test_df = train_test_split_by_players(players_filtered_matches_df['player'].unique(), players_filtered_matches_df, test_size)

        games_filtered_train_df_w = pd.merge(games_filtered, train_df[['player', 'game_index']], left_on=['winner', 'game_index'], right_on=['player', 'game_index'])
        games_filtered_train_df_l = pd.merge(games_filtered, train_df[['player', 'game_index']], left_on=['loser', 'game_index'], right_on=['player', 'game_index'])
        games_filtered_train_df = pd.concat([games_filtered_train_df_w, games_filtered_train_df_l]).sort_values('timestamp')

        games_filtered_test_df_w = pd.merge(games_filtered, test_df[['player', 'game_index']], left_on=['winner', 'game_index'], right_on=['player', 'game_index'])
        games_filtered_test_df_l = pd.merge(games_filtered, test_df[['player', 'game_index']], left_on=['loser', 'game_index'], right_on=['player', 'game_index'])
        games_filtered_test_df = pd.concat([games_filtered_test_df_w, games_filtered_test_df_l]).sort_values('timestamp')

        # filter out games appearing in both test and train set
        # it could happen that player A's last X% of games in the test set occur in player B's first 1-X% games in the train set
        games_filtered_test_unique_df = games_filtered_test_df[~games_filtered_test_df.game_index.isin(games_filtered_train_df.game_index.tolist())]

        return games_filtered_train_df, games_filtered_test_unique_df
    
#we don't do train_test_split for UFC data at the moment since it has only 7660 games
#the calibration plot on the entire 7660 games dataset is very similar to the calibration plot of the entire 25651 boxing data
#class UFCSpliter(DataSpliter)