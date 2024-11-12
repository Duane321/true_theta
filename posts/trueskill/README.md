# TrueSkill for the True Theta Blog (parts 1 and 2) and Mutual Information Video

This repository contains the code for the True Theta blog posts on TrueSkill and the Mutual Information video. Both are
currently in progress and will be released soon.

A description of the primary files.

- `true_skill_simulated.ipynb`: A notebook that creates the graphics for the Part 1 blog post on TrueSkill. It still needs
  to be cleaned up a bit.
- `trueskill_simulation.py`: A primary class used in the true_skill_simulated.ipynb notebook.
- `true_skill_wc3.ipynb`: This notebook applies [TrueSkillThroughTime](https://github.com/glandfried/TrueSkillThroughTime.py) to warcraft3.
- `true_skill_wc3_dev.ipynb`: This basically does exactly the same thing as `true_skill_wc3.ipynb`, but it does it via
class that is in the `true_skill_through_time.py` file.
- `true_skill_through_time.py`: This contains the `TrueSkillThroughTimeApplied` class, which is used in the `true_skill_wc3_dev.ipynb` notebook.
- `true_skill_boxing.ipynb`: This is an incomplete notebook that creates a boxing match dataset from the boxing records
of many fighters. The data is nearly in the right format, but not quite there. See the notebook for more details.