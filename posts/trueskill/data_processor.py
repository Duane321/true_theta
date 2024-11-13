from abc import ABC, abstractmethod
import urllib
import pandas as pd
import numpy as np
from utils import *

class DataProcessor(ABC):
    def __init__(self, input_filename: str):
        self.input_filename = input_filename

    @abstractmethod
    def pull_data(self) -> pd.DataFrame:
        pass

    @abstractmethod
    def process_games_data(self, input_data=None) -> pd.DataFrame:
        pass


class Warcraft3Processor(DataProcessor):
    # TODO - add on how to get data\warcraft3.csv
    def pull_data(self):
        pass

    def process_games_data(self):
        #e.g. 'data\warcraft3.csv'
        games_raw = pd.read_csv(self.input_filename).query('(competitor_1_score > -0.0001) & (competitor_2_score > -0.0001)').iloc[-50000:]
        games_raw['timestamp'] = pd.to_datetime(games_raw['date'])

        game_composition = []
        times = []
        np.random.seed(0)

        for _, row in games_raw.iterrows():
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



class TennisProcessor(DataProcessor):
    def pull_data(self):
        pass

    def process_games_data(self):
        #e.g. "data/tennis_history.csv"
        tennis_data_raw = pd.read_csv(self.input_filename, low_memory=False)
        tennis_data_df = tennis_data_raw[tennis_data_raw.double=='f'][['match_id', 'w1_id', 'l1_id', 'time_end']].dropna()
        tennis_data_df.time_end = pd.to_datetime(tennis_data_df.time_end)
        tennis_data_df = tennis_data_df.rename(columns={'w1_id': 'winner', 'l1_id': 'loser', 'time_end': 'timestamp'}).sort_values('timestamp')
        #e.g output filename: "data/tennis_matches_refined_tstt.parquet"
        return tennis_data_df


class BoxingProcessor(DataProcessor):
    # TODO - do we need to add a method on how to get wiki_urls.txt for boxing and UFC?
    def pull_data(self):
        # data/boxer_wiki_urls.txt contains the wikipedia URLs of a large list of boxers
        with open(self.input_filename, 'r') as file:
            urls = file.readlines()
        urls = [url.strip() for url in urls]

        # it takes about 20mins to pull the raw data from Wikipedia
        boxing_records = {}
        for url in urls:
            print(url)
            boxer_name = url[30:]
            try:
                record = extract_boxing_record(url)
                if record is not None:
                    boxing_records[boxer_name] = record
            except:
                print(f"broke on: {url}")

        boxing_records = {k: v for k, v in boxing_records.items() if all([c in v.columns for c in ['Date raw', 'Result', 'Result']])}

        for k, v in boxing_records.items():
            v['Date'] = parse_dates(v['Date raw'])

        boxing_matches = []
        for k, v in boxing_records.items():
            fighter = urllib.parse.unquote(k.replace('_', ' ').replace('(boxer)', '').strip())

            if fighter == 'Boxing career of Manny Pacquiao':
                fighter = 'Manny Pacquiao'

            v = v.rename(columns={'Res.':'Result'})
            logi = v['Date'].isnull()
            print(f"Dropping: {logi.sum()}")
            
            boxing_matches.append(v[~logi].assign(Fighter=fighter)[['Fighter', 'Opponent', 'Result', 'Date', 'Date raw']])

        boxing_matches_df = pd.concat(boxing_matches, axis=0).reset_index(drop=True)

        return boxing_matches_df

    def process_games_data(self, boxing_matches_df):
        
        boxing_matches_df = boxing_matches_df[boxing_matches_df['Result'].isin(['Loss', 'Lost', 'L by TKO', 'L by KO'
                                , 'Lose', 'LOST', 'Wim', 'Win', 'Won', 'W by KO', 'W by TKO', 'W by SD', 'W by PTS'])]

        mapper = {'Loss':0, 'Lost':0, 'L by TKO':0, 'L by KO':0, 'Lose':0, 'LOST':0,
          'Win':1, 'Wim':1, 'Won':1, 'W by KO':1, 'W by TKO':1, 'W by SD':1, 'W by PTS':1}
        
        boxing_matches_df['Outcome'] = boxing_matches_df['Result'].apply(lambda x: mapper[x])
        #Sometimes, a fighter's name will appear differently on their own page than on another fighters page. So we have to map them to be the same string. We do this by guessing which names are likely misspellings of each (by counting overlaps in their character histogram). Then we map the names with a manually created dictionary.
        name_changes = {}
        name_changes["Rey Migreno"] = "Rey Megrino"
        name_changes["George Ashe"] = "George Ashie"
        name_changes["Stanyslav Tomkachov"] = "Stanyslav Tovkachov"
        name_changes["Greg Scott-Briggs"] = "Greg Scott Briggs"
        name_changes["Kongthawat Sorkitti"] = "Kongthawat Sor Kitti"
        name_changes["Rogelio Castañeda"] = "Rogelio Castaneda"
        name_changes["Miguel Angel Suarez"] = "Miguel Angel Saurez"
        name_changes["Nikolay Eremeev"] = "Nikolay Emereev"
        name_changes["Bill Haderman"] = "Bill Hardeman"
        name_changes["Rubén Darío Palacios"] = "Rubén Darío Palacio"
        name_changes["George Kambosos Jr."] = "George Kambosos Jr"
        name_changes["Sven Erik Paulsen"] = "Svein Erik Paulsen"
        name_changes["Singnum Chuwatana"] = "Singnum Chuwattana"
        name_changes["Mohammed Medjadi	"] = "Mohammed Medjadji"

        for c in ['Fighter', 'Opponent']:
            boxing_matches_df[c] = boxing_matches_df[c].apply(lambda x: name_changes[x] if x in name_changes else x)
            boxing_matches_df[c] = boxing_matches_df[c].str.replace('Jr.', 'Jr')
            boxing_matches_df[c] = boxing_matches_df[c].str.replace('Sr.', 'Sr')

        boxing_matches_df['key'] = np.nan

        for i, row in boxing_matches_df.iterrows():
            f, o, d = row['Fighter'], row['Opponent'], row['Date']
            if f < o:
                k = f'{f}_{o}_{d}'
            else:
                k = f'{o}_{f}_{d}'
            boxing_matches_df.loc[i, 'key'] = k

        boxing_matches_df = boxing_matches_df.drop_duplicates(subset=['key']).sort_values(['Date', 'Fighter']).reset_index(drop=True)
        boxing_matches_df['Winner'] = boxing_matches_df.apply(lambda row: row.Fighter if row.Outcome==1 else row.Opponent, axis=1)
        boxing_matches_df['Loser'] = boxing_matches_df.apply(lambda row: row.Opponent if row.Outcome==1 else row.Fighter, axis=1)
        boxing_matches_df = boxing_matches_df.rename(columns={'Winner': 'winner', 'Loser': 'loser', 'Date': 'timestamp'})
        boxing_matches_df = boxing_matches_df.reindex(columns=['winner', 'loser', 'timestamp'])

        return boxing_matches_df




class UFCProcessor(DataProcessor):
    def pull_data(self):
        # data/ufc_wiki_urls_v2.txt contains the wikipedia URLs of a large list of ufc fighters
        with open(self.input_filename, 'r') as file:
            urls = file.readlines()
        urls = [url.strip() for url in urls]

        ufc_records = {}

        for url in urls[:]:
            print(url)
            fighter_name = url[30:]
            print('fighter_name:', fighter_name)
            try:
                record = extract_ufc_record(url)
                if record is not None:
                    ufc_records[fighter_name] = record
            except:
                print(f"broke on: {url}")

        ufc_records = {k: v for k, v in ufc_records.items() if all([c in v.columns for c in ['Date raw', 'Result', 'Result']])}

        for k, v in ufc_records.items():
            v['Date'] = parse_dates(v['Date raw'])

        ufc_matches = []
        for k, v in ufc_records.items():
            fighter = urllib.parse.unquote(k.replace('_', ' ').replace('(fighter)', '').strip())
            logi = v['Date'].isnull()
            print(f"Dropping: {logi.sum()}")
            
            ufc_matches.append(v[~logi].assign(Fighter=fighter)[['Fighter', 'Opponent', 'Result', 'Date', 'Date raw']])

        ufc_matches_df = pd.concat(ufc_matches, axis=0).reset_index(drop=True)
        
        return ufc_matches_df



    def process_games_data(self, ufc_matches_df):
        
        ufc_matches_df = ufc_matches_df[ufc_matches_df['Result'].isin(['Win', 'Loss'])]
        mapper = {'Loss':0, 
          'Win':1}
        ufc_matches_df['Outcome'] = ufc_matches_df['Result'].apply(lambda x: mapper[x])
        
        #Sometimes, a fighter's name will appear differently on their own page than on another fighters page. So we have to map them to be the same string. We do this by guessing which names are likely misspellings of each (by counting overlaps in their character histogram). Then we map the names with a manually created dictionary.
        name_changes = {}
        name_changes["Baasankhuu Damnlanpurev"] = "Baasankhuu Damlanpurev"
        name_changes["Abdulhalik Magomedov"] = "Abdulkhalik Magomedov"
        name_changes["Gaetano Pirello"] = "Gaetano Pirrello"
        name_changes["Rafael Correa"] = "Rafael Correia"
        name_changes["Tyler Bialeck"] = "Tyler Bialecki"
        name_changes["Dave Moran"] = "Dave Morgan"
        name_changes["Benoît Saint Denis"] = "Benoît Saint-Denis"
        name_changes["Isabela de Padua"] = "Isabela de Pádua"
        name_changes["Gilberto Galvao"] = "Gilberto Galvão"
        name_changes["Piera Rodriguez"] = "Piera Rodríguez"
        name_changes["Adrian Yanez"] = "Adrian Yañez"
        name_changes["Diego Lopez"] = "Diego Lopes"

        for c in ['Fighter', 'Opponent']:
            ufc_matches_df[c] = ufc_matches_df[c].apply(lambda x: name_changes[x] if x in name_changes else x)
            ufc_matches_df[c] = ufc_matches_df[c].str.replace('Jr.', 'Jr')
            ufc_matches_df[c] = ufc_matches_df[c].str.replace('Sr.', 'Sr')

        ufc_matches_df['key'] = np.nan

        for i, row in ufc_matches_df.iterrows():
            f, o, d = row['Fighter'], row['Opponent'], row['Date']
            if f < o:
                k = f'{f}_{o}_{d}'
            else:
                k = f'{o}_{f}_{d}'
            ufc_matches_df.loc[i, 'key'] = k

        ufc_matches_df = ufc_matches_df.drop_duplicates(subset=['key']).sort_values(['Date', 'Fighter']).reset_index(drop=True)
        ufc_matches_df['Winner'] = ufc_matches_df.apply(lambda row: row.Fighter if row.Outcome==1 else row.Opponent, axis=1)
        ufc_matches_df['Loser'] = ufc_matches_df.apply(lambda row: row.Opponent if row.Outcome==1 else row.Fighter, axis=1)

        ufc_matches_df = ufc_matches_df.rename(columns={'Winner': 'winner', 'Loser': 'loser', 'Date': 'timestamp'})
        ufc_matches_df = ufc_matches_df.reindex(columns=['winner', 'loser', 'timestamp'])

        return ufc_matches_df
