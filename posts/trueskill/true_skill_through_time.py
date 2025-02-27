import pandas as pd
import numpy as np
import trueskillthroughtime as tst
import altair as alt
from scipy.optimize import minimize
from scipy.stats import norm


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
            self.make_game_composition(),
            times=self.games["time_0_to_999_int"].tolist(),
            gamma=gamma,
            sigma=sigma,
            beta=beta,
        )

    def negative_log_evidence(self, params):
        gamma, sigma, beta = params
        print(f"\ngamma: {gamma:.4f}, sigma: {gamma:.4f}, beta: {gamma:.4f}, ")
        history = self.get_history(gamma=gamma, sigma=sigma, beta=beta)
        nle = -history.log_evidence()
        print(f"NLE: {nle:.4f}")
        return nle

    def learn_optimal_parameters(self, game_type: str):

        # For Warcraft 3, Boxing, UFC Initial guesses for gamma and sigma
        # initial_params = [.02, 5.0, 1]

        # initial guess of the params, for tennis we have a lower sigma
        initial_params = [0.02, 1.5, 1] if game_type == "tennis" else [0.02, 5.0, 1]

        if game_type == "tennis":
            bounds = [
                (0.001, 0.05),  # gamma: tighter upper bound for stability
                (0.5, 4.0),  # sigma: narrower range for better convergence
                (0.5, 4.0),  # beta: matched to sigma for balanced scaling
            ]
        elif game_type == "warcraft3":
            bounds = [(0.00001, 0.1), (0.0001, 8), (0.0001, 8)]
        else:
            # set a larger lower bound for each param to ensure numerical stability for Boxing and UFC
            bounds = [(0.001, 0.1), (0.01, 8), (0.01, 8)]

        result = minimize(self.negative_log_evidence, initial_params, bounds=bounds)
        self.gamma_optimal, self.sigma_optimal, self.beta_optimal = result.x

        print(f"optimal gamma : {self.gamma_optimal:.4f}")
        print(f"optimal sigma : {self.sigma_optimal:.4f}")
        print(f"optimal beta : {self.beta_optimal:.4f}")

    def set_optimal_parameters(self, gamma: float, sigma: float, beta: float):
        self.gamma_optimal = gamma
        self.sigma_optimal = sigma
        self.beta_optimal = beta

    def set_skill_curves(self):

        history = self.get_history(
            gamma=self.gamma_optimal, sigma=self.sigma_optimal, beta=self.beta_optimal
        )
        learning_curves = history.learning_curves()
        self.skill_curves = {
            p: learning_curves[self.competitor_name_map[p]] for p in self.competitors
        }
        return self.skill_curves

    def plot_player_skills(
        self,
        players: list[str],
        players_id_mapping=None,
        width: int = 1000,
        height: int = 800,
        burnin: int = 20,
        x_axis_format="%Y %B",
        y_lims=None,
        game: str = None,
    ):

        assert (
            self.skill_curves is not None
        ), "Skill curves not set. Run .set_skill_curves() first."

        t_ints_to_date = (
            self.games.groupby("time_0_to_999_int")["timestamp"]
            .mean()
            .dt.date.to_dict()
        )

        # Create a dataframe for each player and combine them
        df_list = []
        for player, data in self.skill_curves.items():
            data_ = data[burnin:]
            if player not in players:
                continue
            player = players_id_mapping[player] if players_id_mapping else player
            df = pd.DataFrame(
                {
                    "date": [t_ints_to_date[t] for t, _ in data_],
                    "mu": [d.mu for _, d in data_],
                    "sigma": [d.sigma for _, d in data_],
                }
            )
            df["player"] = player
            df["lower"] = df["mu"] - 1 * df["sigma"]
            df["upper"] = df["mu"] + 1 * df["sigma"]
            df_list.append(df)

        df_combined = pd.concat(df_list)
        df_combined["date"] = pd.to_datetime(df_combined["date"])

        y_scale = (
            alt.Scale(zero=False)
            if y_lims is None
            else alt.Scale(domain=y_lims, nice=False)
        )

        line = (
            alt.Chart(df_combined)
            .mark_line(strokeWidth=4)
            .encode(
                x=alt.X("date:T", axis=alt.Axis(format=x_axis_format)),
                y=alt.Y("mu", scale=y_scale, title="Skill"),
                color=alt.Color(
                    "player", scale=alt.Scale(scheme="tableau10"), sort=players
                ),  # , scale=alt.Scale(scheme='plasma')
            )
        )
        confidence_interval = (
            alt.Chart(df_combined)
            .mark_area(opacity=0.4)
            .encode(x="date", y="lower", y2="upper", color="player")
        )

        chart = confidence_interval + line
        return chart.properties(
            width=width,
            height=height,
            title="Skill estimates with one standard deviation uncertainties"
            + (f" for {game}" if game else ""),
        ).configure_axis(grid=False)

    def plot_calibration(self, width: int = 400, height: int = 400):

        curves_map = {k: {t: n for t, n in v} for k, v in self.skill_curves.items()}

        df = []
        for i, row in self.games.iterrows():
            winner, loser, t_int = row["winner"], row["loser"], row["time_0_to_999_int"]
            # avoid overfitting - randomly assigns the winner to c1 and the loser to c2
            if np.random.uniform() < 0.5:
                c1_win = 1
                c1, c2 = winner, loser
            else:
                c1_win = 0
                c1, c2 = loser, winner

            if c1 in curves_map and c2 in curves_map:
                if t_int in curves_map[c1] and t_int in curves_map[c2]:
                    normal_1, normal_2 = curves_map[c1][t_int], curves_map[c2][t_int]
                    mu_diff = normal_1.mu - normal_2.mu
                    sigma2_diff = (
                        normal_1.sigma**2
                        + normal_2.sigma**2
                        + 2 * (self.beta_optimal**2)
                    )
                    # c1_win_prob, _ = quad(normal_pdf, 0, np.inf, args=(mu_diff, sigma2_diff ** .5))
                    # use norm.cdf to speed up the prob calculation, P(X > 0) = 1 - P(X ≤ 0)
                    c1_win_prob = 1 - norm.cdf(0, mu_diff, sigma2_diff**0.5)

                    df.append([c1, c2, c1_win, c1_win_prob])

        df = pd.DataFrame(
            df, columns=["competitor_1", "competitor_2", "win_actual", "win_prob"]
        ).dropna()

        df["win_prob_bucket"] = pd.qcut(df["win_prob"], q=10, duplicates="drop")

        bucket_means = df.groupby("win_prob_bucket").agg(
            avg_outcome=("win_actual", "mean"), win_prob_midpoint=("win_prob", "mean")
        )

        line_plot = (
            alt.Chart(bucket_means)
            .mark_line(strokeWidth=4)
            .encode(
                x=alt.X("win_prob_midpoint", title="Predicted Win Probability"),
                y=alt.Y("avg_outcome", title="Average Outcome"),
            )
        )
        point_plot = (
            alt.Chart(bucket_means)
            .mark_point(fill="white", strokeWidth=4, size=200, opacity=1)
            .encode(
                x=alt.X("win_prob_midpoint"),
                y=alt.Y("avg_outcome"),
            )
        )

        straight_line_data = pd.DataFrame({"x": [0, 1], "y": [0, 1]})

        # Create the line chart with dotted line style
        straight_line = (
            alt.Chart(straight_line_data)
            .mark_line(strokeDash=[12, 12], color="black")
            .encode(x="x", y="y")
        )

        return (straight_line + line_plot + point_plot).properties(
            width=width, height=height, title="Calibration Plot In-Sample"
        )

    def plot_calibration_oos(
        self, oos_data, width: int = 400, height: int = 400, game=None
    ):
        """
        plot calibration for out-of-sample data,
        for each player in the oos data, take the player's last available mu and sigma then compute win_prob
        """
        # {'player_name': [(166, N(mu=2.302, sigma=1.017)), (196, N(mu=2.155, sigma=0.877))]} as in the skill_curves
        last_curves_map = {k: v[-1][1] for k, v in self.skill_curves.items()}

        df = []
        for _, row in oos_data.iterrows():
            winner, loser = row["winner"], row["loser"]
            # avoid overfitting - randomly assigns the winner to c1 and the loser to c2
            if np.random.uniform() < 0.5:
                c1_win = 1
                c1, c2 = winner, loser
            else:
                c1_win = 0
                c1, c2 = loser, winner

            # later TODO - for those that is not on the last_curves_map, we may assume N(0, 1)
            if c1 in last_curves_map and c2 in last_curves_map:
                normal_1, normal_2 = last_curves_map[c1], last_curves_map[c2]
                mu_diff = normal_1.mu - normal_2.mu
                sigma2_diff = (
                    normal_1.sigma**2 + normal_2.sigma**2 + 2 * (self.beta_optimal**2)
                )
                # use norm.cdf to speed up the prob calculation, P(X > 0) = 1 - P(X ≤ 0)
                c1_win_prob = 1 - norm.cdf(0, mu_diff, sigma2_diff**0.5)

                df.append([c1, c2, c1_win, c1_win_prob])

        df = pd.DataFrame(
            df, columns=["competitor_1", "competitor_2", "win_actual", "win_prob"]
        ).dropna()

        df["win_prob_bucket"] = pd.qcut(df["win_prob"], q=10, duplicates="drop")

        bucket_means = df.groupby("win_prob_bucket").agg(
            avg_outcome=("win_actual", "mean"), win_prob_midpoint=("win_prob", "mean")
        )

        line_plot = (
            alt.Chart(bucket_means)
            .mark_line(strokeWidth=4)
            .encode(
                x=alt.X("win_prob_midpoint", title="Predicted Win Probability"),
                y=alt.Y("avg_outcome", title="Average Outcome"),
            )
        )
        point_plot = (
            alt.Chart(bucket_means)
            .mark_point(fill="white", strokeWidth=4, size=200, opacity=1)
            .encode(
                x=alt.X("win_prob_midpoint"),
                y=alt.Y("avg_outcome"),
            )
        )

        straight_line_data = pd.DataFrame({"x": [0, 1], "y": [0, 1]})

        # Create the line chart with dotted line style
        straight_line = (
            alt.Chart(straight_line_data)
            .mark_line(strokeDash=[12, 12], color="black")
            .encode(x="x", y="y")
        )

        return (straight_line + line_plot + point_plot).properties(
            width=width,
            height=height,
            title="Out-Of-Sample Calibration Plot" + (f" for {game}" if game else ""),
        )
