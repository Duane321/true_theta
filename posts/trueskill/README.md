# TrueSkill for the True Theta Blog, Parts 1 and 2.

This repository contains the code for the True Theta blog posts on TrueSkill and the Mutual Information video. Both are
currently in progress and will be released soon.

A description of the primary files. You may run each cell in the notebook one by one to replicate the results.

- `true_skill_simulated.ipynb`: A notebook that creates the graphics for the Part 1 blog post on TrueSkill. 
- `trueskill_simulation.py`: A primary class used in the true_skill_simulated.ipynb notebook.
- `utils.py`: This contains the utility functions, which are used in the application notebook for each game.
- `train_test_split_games.py`: This contains the `DataSplitter` class, which is used to do train-test split in the application notebook for each game.
- `true_skill_through_time.py`: This contains the `TrueSkillThroughTimeApplied` class, which is used to apply the TrueSkillThroughTime algo in the application notebook for each game.

- `warcraft3/warcraft3_dev.ipynb`: This notebook applies [TrueSkillThroughTime](https://github.com/glandfried/TrueSkillThroughTime.py) to warcraft3.
  - It runs the entire pipeline to preprocess the data/warcraft3.csv raw data, do a train-test split based on each player's career path and applies algo in the `true_skill_through_time.py` file. The results are validated via players' skill curves through time, calibration plots of train and test sets and the AUC.

- `tennis/tennis_dev.ipynb`: This notebook applies [TrueSkillThroughTime](https://github.com/glandfried/TrueSkillThroughTime.py) to tennis.
  - It runs the entire pipeline to preprocess the data/tennis_history.csv raw data(pulled from the TrueSkillThroughTime github documentation above), do a train-test split based on each player's career path and applies algo in the `true_skill_through_time.py` file. The results are validated via players' skill curves through time, calibration plots of train and test sets and the AUC.

- `boxing/boxing_dev.ipynb`: This notebook applies [TrueSkillThroughTime](https://github.com/glandfried/TrueSkillThroughTime.py) to boxing.
  - It runs the entire pipeline to first preprocess the data/boxer_wiki_urls.txt to fetch game history from each famous player's Wiki page, do manual corrections and de-duplications in order to have a well-formed game dataset. Then it does a train-test split based on each player's career path and applies algo in the `true_skill_through_time.py` file. The results are validated via players' skill curves through time, calibration plots of train and test sets and the AUC.

- `data`: This folder contains raw and formatted data for each game.
  - `boxer_wiki_urls.txt`: each line has a famous boxer's url
  - `boxing_matches_cleaned.parquet`: boxing match history in winner-loser-timestamp format after some clean-up
  - `players_ge_40_matches_lst_boxing.json`: a list of boxers who have played at least 40 games in the career
  - `warcraft3.csv`: raw game history of warcraft3
  - `players_ge_40_matches_lst_warcraft3.json`: a list of warcraft3 players who have played at least 40 games in the career
  - `tennis_history.csv`: raw game history of tennis
  - `tennis_players_ge_40_matches_lst.json`: a list of tennis players who have played at least 40 games in the career
  - `tennis_player_id_map.json`: a dictionary of tennis player's id to player's name
  - `tennis_player_id_inv_map.json`: a dictionary of tennis player's name to player's id
  - `tennis_matches_refined_tstt.parquet`: tennis match history in winner-loser-timestamp format after some clean-up

  Basic Steps for Data Ingestion and Preprocessing:
  - Collecting game history for players in the game
  - Handling different match formats
  - Dealing with missing or incomplete data
  - Converting to consistent winner-loser-timestamp format

  Overview of datasets sizes
  - Tennis: 326,306 matches, 2502 players who have played at least 40 games in the career
  - Warcraft 3: 85,777 matches, 175 players who have played at least 40 games in the career
  - Boxing: 25,651 matches, 303 players who have played at least 40 games in the career

- `plots_updated_params`: plots of latest model. For each game, it contains skillcurve of major players, calibration of train and test set. 
