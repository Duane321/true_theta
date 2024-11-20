# TrueSkill for the True Theta Blog (parts 1 and 2) and Mutual Information Video

This repository contains the code for the True Theta blog posts on TrueSkill and the Mutual Information video. Both are
currently in progress and will be released soon.

A description of the primary files. You may run each cell in the notebook one by one to replicate the results.

- `true_skill_simulated.ipynb`: A notebook that creates the graphics for the Part 1 blog post on TrueSkill. It still needs to be cleaned up a bit.
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

- `ufc/ufc_dev.ipynb`: This notebook applies [TrueSkillThroughTime](https://github.com/glandfried/TrueSkillThroughTime.py) to boxing.
  - It runs the entire pipeline to first preprocess the data/ufc_wiki_urls_v2.txt to fetch game history from each famous player's Wiki page, do manual corrections and de-duplications in order to have a well-formed game dataset. Then it applies algo in the `true_skill_through_time.py` file. The results are validated via players' skill curves through time, calibration plots of train and test sets and the AUC.

- `data`: This folder contains raw and formatted data for each game.

- `plots_updated_params`: plots of latest model. For each game, it contains skillcurve of major players, calibration of train and test set. Please note that since UFC has very few games available, we only do a calibration on the entire dataset without doing train-test split.
