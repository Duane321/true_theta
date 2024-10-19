import pandas as pd
import numpy as np
import trueskillthroughtime as tst
import altair as alt
from scipy.optimize import minimize
#from scipy.integrate import quad
from scipy.stats import norm

# Define the normal distribution PDF
def normal_pdf(x, mu, sigma):
    return (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mu) / sigma) ** 2)

class TrueSkillThroughTimeApplied:

    def __init__(self, games: pd.DataFrame):
        """
        Initialize the TrueSkillThroughTimeApplied object with the games dataframe.

        The games dataframe should look like:

                winner         loser           timestamp
            0   TeRRoR           Ark 2020-05-10 00:00:36
            1   Rigety  MisterWinner 2020-05-10 00:00:37
            2  Paladyn        SaturN 2020-05-10 00:00:38
            3    Blade    TimberOwls 2020-05-10 00:00:39
            4     HawK     GodFather 2020-05-10 00:00:40
        """
        self.games = self.check_and_adjust_games(games)
        self.competitors = list(
            set(self.games["winner"].values) | set(self.games["loser"].values)
        )

        # The below shouldn't be necessary, but there is some quirk of the trueskillthroughtime package that requires players are named with one character.
        self.competitor_name_map = {
            c: chr(97 + i) for i, c in enumerate(self.competitors)
        }
        self.competitor_name_map_inv = {
            l: c for c, l in self.competitor_name_map.items()
        }

        self.gamma_optimal = None
        self.sigma_optimal = None
        self.beta_optimal = None

        self.skill_curves = None

    def check_and_adjust_games(self, games: pd.DataFrame):
        """
        Check if the games dataframe is in the correct format and append columns.
        """

        for c in ["winner", "loser", "timestamp"]:
            assert c in games.columns, f"{c} not in games columns"

        assert games["timestamp"].dtype == np.dtype(
            "<M8[ns]"
        ), "timestamp not in datetime format"
        assert games[
            "timestamp"
        ].is_monotonic_increasing, "timestamp not in increasing order"
        assert games.shape[0] > 0, "no games"

        games["time_0_to_999_int"] = (
            (games["timestamp"] - games["timestamp"].min())
            / (games["timestamp"].max() - games["timestamp"].min())
            * 999
        ).astype(int)

        return games

    def get_top_k_most_playing_players(self, k):
        games_played = (
            (
                self.games["winner"].value_counts().reindex(self.competitors).fillna(0)
                + self.games["loser"].value_counts().reindex(self.competitors).fillna(0)
            )
            .sort_values(ascending=False)
            .astype(int)
        )
        return games_played, games_played.index[:k].values

    def make_game_composition(self):
        return [
            [self.competitor_name_map[vi] for vi in v]
            for v in self.games[["winner", "loser"]].values
        ]

    def get_history(self, gamma, sigma, beta):
        return tst.History(
            self.make_game_composition(), times=self.games["time_0_to_999_int"].tolist(), gamma=gamma, sigma=sigma, beta=beta
        )

    def negative_log_evidence(self, params):
        gamma, sigma, beta = params
        print(f"\ngamma: {gamma:.4f}, sigma: {gamma:.4f}, beta: {gamma:.4f}, ")
        history = self.get_history(gamma=gamma, sigma=sigma, beta=beta)
        nle = -history.log_evidence()
        print(f"NLE: {nle:.4f}")
        return nle

    def learn_optimal_parameters(self):

        # Initial guesses for gamma and sigma
        initial_params = [.02, 5.0, 1]

        # Define the bounds for both gamma and sigma (in this case, between 0 and 10 for each)
        #bounds = [(0.00001, 0.1), (0.0001, 8), (0.0001, 8)]

        # set a larger lower bound for each param to ensure numerical stability(otherwise it won't converge)
        bounds = [(0.001, 0.1), (0.01, 8), (0.01, 8)]

        # Perform the optimization with bounds
        result = minimize(self.negative_log_evidence, initial_params, bounds=bounds)

        # Print the result
        self.gamma_optimal, self.sigma_optimal, self.beta_optimal = result.x

        print(f"optimal gamma : {self.gamma_optimal:.4f}")
        print(f"optimal sigma : {self.sigma_optimal:.4f}")
        print(f"optimal beta : {self.beta_optimal:.4f}")

    def set_optimal_parameters(self, gamma: float, sigma: float, beta: float):
        self.gamma_optimal = gamma
        self.sigma_optimal = sigma
        self.beta_optimal = beta

    def set_skill_curves(self):

        history = self.get_history(gamma=self.gamma_optimal, sigma=self.sigma_optimal, beta=self.beta_optimal)
        learning_curves = history.learning_curves()
        self.skill_curves = {p: learning_curves[self.competitor_name_map[p]] for p in self.competitors}
        return self.skill_curves

    def plot_player_skills(self, players:list[str], width:int=1000, height:int=800, burnin:int=20):

        assert self.skill_curves is not None, "Skill curves not set. Run .set_skill_curves() first."

        t_ints_to_date = self.games.groupby('time_0_to_999_int')['timestamp'].mean().dt.date.to_dict()

        # Create a dataframe for each player and combine them
        df_list = []
        for player, data in self.skill_curves.items():
            data_ = data[burnin:]
            if player not in players:
                continue
            df = pd.DataFrame({
                'date': [t_ints_to_date[t] for t, _ in data_],
                'mu': [d.mu for _, d in data_],
                'sigma': [d.sigma for _, d in data_]
            })
            df['player'] = player
            df['lower'] = df['mu'] - 1 * df['sigma']
            df['upper'] = df['mu'] + 1 * df['sigma']
            df_list.append(df)

        df_combined = pd.concat(df_list)
        df_combined["date"] = pd.to_datetime(df_combined["date"])

        line = alt.Chart(df_combined).mark_line(strokeWidth=4).encode(
            x=alt.X('date:T', axis=alt.Axis(format="%Y %B")),
            y=alt.Y('mu', scale=alt.Scale(zero=False), title='Skill'),
            color=alt.Color('player', sort=players)
        )
        confidence_interval = alt.Chart(df_combined).mark_area(opacity=0.4).encode(
            x='date',
            y='lower',
            y2='upper',
            color='player'
        )

        chart = confidence_interval + line
        return chart.properties(width=width, height=height, title='Skills over Time with 1 Std bands').configure_axis(grid=False)

    def plot_calibration(self, width: int = 400, height: int = 400):

        curves_map = {k: {t: n for t, n in v} for k, v in self.skill_curves.items()}

        df = []
        for i, row in self.games.iterrows():
            winner, loser, t_int = row['winner'], row['loser'], row['time_0_to_999_int']
            if np.random.uniform() < .5:
                c1_win = 1
                c1, c2 = winner, loser
            else:
                c1_win = 0
                c1, c2 = loser, winner

            if c1 in curves_map and c2 in curves_map:
                if t_int in curves_map[c1] and t_int in curves_map[c2]:
                    normal_1, normal_2 = curves_map[c1][t_int], curves_map[c2][t_int]
                    mu_diff = normal_1.mu - normal_2.mu
                    sigma2_diff = normal_1.sigma ** 2 + normal_2.sigma ** 2 + 2 * (self.beta_optimal ** 2)
                    #c1_win_prob, _ = quad(normal_pdf, 0, np.inf, args=(mu_diff, sigma2_diff ** .5))
                    #use norm.cdf to speed up the prob calculation, P(X > 0) = 1 - P(X ≤ 0)
                    c1_win_prob = 1 - norm.cdf(0, mu_diff, sigma2_diff ** .5)

                    df.append([c1, c2, c1_win, c1_win_prob])

        df = pd.DataFrame(df, columns=['competitor_1', 'competitor_2', 'win_actual', 'win_prob']).dropna()

        df['win_prob_bucket'] = pd.qcut(df['win_prob'], q=10, duplicates='drop')

        bucket_means = df.groupby('win_prob_bucket').agg(
            avg_outcome=('win_actual', 'mean'),
            win_prob_midpoint=('win_prob', 'mean')
        )

        line_plot = alt.Chart(bucket_means).mark_line(strokeWidth=4).encode(
            x=alt.X('win_prob_midpoint', title='Predicted Win Probability'),
            y=alt.Y('avg_outcome', title='Average Outcome'),
        )
        point_plot = alt.Chart(bucket_means).mark_point(fill='white', strokeWidth=4, size=200, opacity=1).encode(
            x=alt.X('win_prob_midpoint'),
            y=alt.Y('avg_outcome'),
        )

        straight_line_data = pd.DataFrame({
            'x': [0, 1],
            'y': [0, 1]
        })

        # Create the line chart with dotted line style
        straight_line = alt.Chart(straight_line_data).mark_line(strokeDash=[12, 12], color='black').encode(
            x='x',
            y='y'
        )

        return (straight_line + line_plot + point_plot).properties(width=width, height=height, title='Calibration Plot')




