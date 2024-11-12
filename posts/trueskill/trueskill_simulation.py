import pandas as pd
import numpy as np
import trueskill as ts
import altair as alt
from scipy.stats import norm


class TrueskillSimulation:

    """
    This is the True Skill simulation used for the True SKill Part 1 blog.

    See the true_skill_simulated.ipynb notebook for an example of how to use this class.
    """
    
    def __init__(
        self,
        N=18,
        M=1200,
        epsilon=0.01,
        gamma=0.03,
        beta=0.25,
        mu_0=15,
        sigma_0=0.1,
        skill_cheats_for_identifiability=1,
    ):
        """
        N: Number of players. It's good to use a multiple of 6, to avoid uneven teams.
        M: Number of matches, which is also the number of timestamps.
        gamma: Standard deviation of skill changes between games.
        beta: Standard deviation of a game's performance around a skill.
        mu_0: Aprior mean skill.
        sigma_0: Aprior standard deviation of skill.
        skill_cheats_for_identifiability: To avoid identifiability issues, we 'cheat' by allowing the model to know the
            true skill of the several players.
        """
        self.N = N
        self.M = M
        self.epsilon = epsilon
        self.gamma = gamma
        self.beta = beta
        self.mu_0 = mu_0
        self.sigma_0 = sigma_0
        self.skill_cheats_for_identifiability = skill_cheats_for_identifiability

        ts.setup(
            mu=mu_0, sigma=sigma_0, beta=beta, tau=gamma, draw_probability=0.001
        )  # change draw_probability to zero?

        # Data to be simulated
        self.skills = None
        self.performances = None
        self.teams = None
        self.team_performances = None
        self.ranks = None

        # Ratings to be inferred.
        self.ratings = None

    def simulate_data(self):
        """
        Simulate the data.
        """

        # For convenience, we assume all players play exactly once, at each t.
        skills = np.random.normal(
            loc=self.mu_0, scale=self.sigma_0, size=(self.N, 1)
        ) + np.random.normal(loc=0, scale=self.gamma, size=(self.N, self.M)).cumsum(
            axis=1
        )
        performances = np.random.normal(loc=skills, scale=self.beta)

        # Team assignments are random. It's 50% chance of a free-for-all, 25% chance teams of 2 and 25% chance teams of 3.
        teams = np.zeros((self.N, self.M), dtype=int)
        for t in range(self.M):
            u = np.random.uniform()
            if u < 0.5:
                teams[:, t] = np.arange(self.N)  # Free for all
            elif u < 0.75:
                teams[:, t] = np.arange(self.N).repeat(2)[: self.N]  # Teams of 2
            else:
                teams[:, t] = np.arange(self.N).repeat(3)[: self.N]  # Teams of 3

        team_performances = []
        ranks = []

        for t in range(self.M):
            performances_t = performances[:, t]
            teams_t = teams[:, t]

            team_performances_t = {}

            for p, team in enumerate(teams_t):
                if team in team_performances_t:
                    team_performances_t[team] += performances_t[p]
                else:
                    team_performances_t[team] = performances_t[p]

            ranks_t = sorted(team_performances_t, key=lambda x: -team_performances_t[x])

            team_performances.append(team_performances_t)
            ranks.append(ranks_t)

        self.skills = skills
        self.performances = performances
        self.teams = teams
        self.team_performances = team_performances
        self.ranks = ranks

    def apply_true_skill(self):
        """
        Apply the TrueSkill algorithm to the simulated data, which means creating the self.ratings attribute.
        """

        ratings = []
        ratings_t = [ts.Rating() for _ in range(self.N)]

        for t in range(self.M):

            teams_t = self.teams[:, t]

            rating_groups = [{} for _ in range(max(teams_t) + 1)]
            for p, team in enumerate(teams_t):
                rating_groups[team][p] = ratings_t[p]

            rating_groups = [rating_groups[r] for r in self.ranks[t]]
            rating_groups_updated = ts.rate(rating_groups)
            rating_groups_updated_flat = {
                k: v for team in rating_groups_updated for k, v in team.items()
            }
            ratings_t_updated = [rating_groups_updated_flat[i] for i in range(self.N)]

            # Cheating to handle non-identifiability
            for c in range(self.skill_cheats_for_identifiability):
                ratings_t_updated[0] = ts.Rating(
                    mu=self.skills[0, t], sigma=ratings_t_updated[0].sigma
                )

            ratings.append(ratings_t_updated)
            ratings_t = ratings_t_updated

        self.ratings = ratings

    def get_estimated_skill_df(self, player_i: int):
        skill_true = self.skills[player_i]
        performance = self.performances[player_i]
        skill_mu = [rat[player_i].mu for rat in self.ratings]
        skill_sigma = [rat[player_i].sigma for rat in self.ratings]
        return pd.DataFrame(
            dict(
                skill_true=skill_true,
                performance=performance,
                skill_mu=skill_mu,
                skill_sigma=skill_sigma,
            )
        ).assign(z_error=(skill_true - skill_mu) / skill_sigma)

    def plot_skills(self, first_k: int):
        df = pd.DataFrame(self.skills[:first_k, :])
        df = df.reset_index().melt(
            id_vars="index", var_name="m", value_name="skill_value"
        )
        return (
            alt.Chart(df)
            .mark_line()
            .encode(
                x=alt.X("m:Q", title="Match (m)"),
                y=alt.Y("skill_value:Q", title="Skill", scale=alt.Scale(zero=False)),
                color=alt.Color("index:N", title="Player"),
            )
            .properties(
                title=f"Skills of {first_k} players over {self.M} Matches",
                width=700,
                height=300,
            )
            .configure_axis(grid=False)
        )

    def plot_teams_and_ranks(self, height, width, square_size=300):
        players = [f"p{i}" for i in range(1, self.N + 1)]
        match_index = pd.Index(range(self.M), name="match")

        team_df = (
            pd.DataFrame(
                self.teams + 1,
                index=pd.Index(players, name="player"),
                columns=match_index,
            )
            .reset_index()
            .melt(id_vars=["player"])
            .assign(team=lambda df: [f"t{v}" for v in df["value"]])
        )

        color_scheme = "dark2"

        chart_teams = (
            alt.Chart(team_df)
            .mark_point(strokeWidth=0, shape="square", size=square_size, opacity=1)
            .encode(
                x=alt.X("match:O", title="Match (m)"),
                y=alt.Y("player:N", sort=players[::-1]),
                fill=alt.Fill(
                    "team:N", title="team", scale=alt.Scale(scheme=color_scheme)
                ),
                color=alt.Color(
                    "team:N", title="team", scale=alt.Scale(scheme=color_scheme)
                ),
            )
            .properties(height=height, width=width, title="Team Assignments")
        )

        team_names = [f"t{i}" for i in range(1, self.N + 1)]
        ranks_df = (
            (
                pd.DataFrame(
                    [r + [np.nan] * (6 - len(r)) for r in self.ranks],
                    index=match_index,
                    columns=pd.Index(team_names, name="team"),
                )
                + 1
            )
            .stack()
            .reset_index()
            .rename(columns={0: "rank"})
        )
        chart_ranks = (
            alt.Chart(ranks_df)
            .mark_point(strokeWidth=0, shape="square", size=square_size, opacity=1)
            .encode(
                x=alt.X("match:O", title="Match (m)"),
                y=alt.Y("rank:N", sort="descending"),
                fill=alt.Fill(
                    "team:N", title="team", scale=alt.Scale(scheme=color_scheme)
                ),
                color=alt.Color(
                    "team:N", title="team", scale=alt.Scale(scheme=color_scheme)
                ),
            )
            .properties(height=height, width=width, title="Team Ranks")
        )

        return chart_teams & chart_ranks

    def plot_2_player_wins(self):
        first_k = 2
        df_skills = pd.DataFrame(self.skills[:first_k, :])
        df_skills = df_skills.reset_index().melt(
            id_vars="index", var_name="m", value_name="skill_value"
        )
        chart_skills = (
            alt.Chart(df_skills)
            .mark_line()
            .encode(
                x=alt.X("m:Q", title="Match (m)"),
                y=alt.Y("skill_value:Q", title="Skill", scale=alt.Scale(zero=False)),
                color=alt.Color("index:N", title="Player"),
            )
            .properties(
                title=f"Skills and Performance of two players over {self.M} matches, with wins in green",
                width=700,
                height=300,
            )
        )

        df_performances = pd.DataFrame(self.performances[:first_k, :])
        df_performances = df_performances.reset_index().melt(
            id_vars="index", var_name="m", value_name="performance"
        )
        chart_performances = (
            alt.Chart(df_performances)
            .mark_point(strokeWidth=3, opacity=1, size=100)
            .encode(
                x=alt.X("m:Q", title="Match (m)"),
                y=alt.Y("performance:Q", title="Skill", scale=alt.Scale(zero=False)),
                color=alt.Color("index:N", title="Player"),
            )
        )
        df_performance_winner = pd.DataFrame(
            {"winner": self.performances[:first_k, :].max(0)}
        ).reset_index()

        chart_winners = (
            alt.Chart(df_performance_winner)
            .mark_point(strokeWidth=0, opacity=1, size=50, fill="lightgreen")
            .encode(
                x=alt.X("index:Q", title="Match (m)"),
                y=alt.Y("winner:Q", title="Skill", scale=alt.Scale(zero=False)),
            )
        )

        return (chart_skills + chart_performances + chart_winners).configure_axis(
            grid=False
        )

    def plot_errors(self, player_i):

        df_p = self.get_estimated_skill_df(player_i=player_i)

        z_error = df_p["z_error"]
        hist, bin_edges = np.histogram(z_error, bins=50, density=True)
        hist_df = pd.DataFrame(
            {"bin_start": bin_edges[:-1], "bin_end": bin_edges[1:], "density": hist}
        ).assign(zero=0)

        x_values = np.linspace(z_error.min(), z_error.max(), 100)
        standard_normal = norm.pdf(x_values, loc=0, scale=1)
        normal_df = pd.DataFrame({"x": x_values, "y": standard_normal})

        histogram = (
            alt.Chart(hist_df)
            .mark_rect(opacity=0.5)
            .encode(
                x=alt.X("bin_start:Q", title="z_error"),
                x2=alt.X2("bin_end:Q", title="z_error"),
                y=alt.Y("zero:Q", title="Density"),
                y2=alt.Y2("density:Q", title="Density"),
            )
            .properties(
                title="Normalized Histogram of z_error with Standard Normal Distribution Overlay"
            )
        )

        normal_distribution = (
            alt.Chart(normal_df)
            .mark_line(color="red")
            .encode(alt.X("x:Q", title="z_error"), alt.Y("y:Q", title="Density"))
        )
        return histogram + normal_distribution

    def plot_player(self, player_i, include_performance=True, width=600, height=400):

        df = self.get_estimated_skill_df(player_i=player_i)
        df["ci_upper"] = df["skill_mu"] + 1.96 * df["skill_sigma"]
        df["ci_lower"] = df["skill_mu"] - 1.96 * df["skill_sigma"]
        line_true = (
            alt.Chart(df.reset_index())
            .mark_line(color="blue")
            .encode(
                x="index:Q",
                y=alt.Y("skill_true:Q", scale=alt.Scale(zero=False), title="Skill"),
            )
        )
        performances_points = (
            alt.Chart(df.reset_index())
            .mark_point(fill="red", opacity=0.5, strokeWidth=0)
            .encode(
                x="index:Q",
                y="performance:Q",
            )
        )
        line_mu = (
            alt.Chart(df.reset_index())
            .mark_line(color="green")
            .encode(
                x=alt.X("index:Q", title="Match (m)"),
                y="skill_mu:Q",
            )
        )

        ci_area = (
            alt.Chart(df.reset_index())
            .mark_area(opacity=0.75, color="lightgreen")
            .encode(x="index:Q", y="ci_lower:Q", y2="ci_upper:Q")
        )
        if include_performance:
            chart = performances_points + ci_area + line_mu + line_true
        else:
            chart = ci_area + line_mu + line_true

        return chart.properties(
            width=width,
            height=height,
            title="The Actual Simulated Skill (Blue) vs Estimated Skill (Green) with 95% Confidence Interval",
        )
