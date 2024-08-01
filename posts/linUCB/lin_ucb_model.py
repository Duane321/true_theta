import numpy as np
import pandas as pd
import altair as alt

X_RANGE = (-1, 1)


class LinUCBSimulator:

    """
    This is the code used to simulate the LinUCB algorithm. It is to be paired with the LinUCB blog post: https://truetheta.io/concepts/reinforcement-learning/lin-ucb/
    """

    def __init__(
        self,
        T: int,
        n_actions: int,
        d_features: int,
        error_std: float | list[float],
        alpha: float,
        lambda_: float = 1,
    ):

        """
        Initialize the LinUCB simulator.

        Parameters:
        -----------
        T: int, number of rounds to simulate
        n_actions: int, number of actions
        d_features: int, number of features
        error_std: float or list of floats, standard deviation of the reward noise
        alpha: float, hyperparameter for the LinUCB algorithm
        lambda_: float, hyperparameter for the LinUCB algorithm
        """

        # Simulation parameters
        self.T = T
        self.n_actions = n_actions
        self.d_features = d_features
        self.error_std = error_std

        # Algorithm hyperparameter
        self.alpha = alpha
        self.lambda_ = lambda_

        self.simulate()
        self.initialize()

    def simulate_X_and_true_theta(self):
        """
        Generates the context and the true theta parameters
        """
        self.X = np.random.uniform(size=(self.T, self.n_actions, self.d_features))
        self.theta_true = np.random.normal(size=(1, self.n_actions, self.d_features))

    def simulate(self):
        """
        Simulates the context, the true theta parameters and the rewards.
        """
        self.simulate_X_and_true_theta()
        self.rewards_mean = (self.X * self.theta_true).sum(-1)
        self.rewards_epsilon = np.random.normal(
            loc=0, scale=self.error_std, size=(self.T, self.n_actions)
        )
        self.rewards = self.rewards_mean + self.rewards_epsilon

    def initialize(self):
        """
        Initialize the parameters of the algorithms.
        """
        self.A = np.array(
            [self.lambda_ * np.eye(self.d_features) for _ in range(self.n_actions)]
        )  # So A[i] will give a diagonal matrix for action i. A[i] will get accumulated with xi * xi^T's
        self.b = np.zeros(
            (self.n_actions, self.d_features)
        )  # b[i] will get accumulated with xi * reward_i's

        self.theta = np.zeros((self.T, self.n_actions, self.d_features))
        self.reward_mean_estimate = np.zeros((self.T, self.n_actions))
        self.uncertainty = np.zeros((self.T, self.n_actions))
        self.P = np.zeros((self.T, self.n_actions))
        self.actions = np.zeros(self.T, dtype=int)

        self.rewards_chosen = np.zeros(self.T)

    def calc_regret(self):
        """
        Calculate the regret and pseudo-regret of the algorithm.
        """

        # Regret calculation
        self.actions_optimal = np.argmax(self.rewards_mean, axis=1)
        self.rewards_optimal = np.array(
            [self.rewards[t, self.actions_optimal[t]] for t in range(self.T)]
        )
        self.regret = np.cumsum(self.rewards_optimal) - np.cumsum(self.rewards_chosen)

        # Psuedo-regret calculation. Similar to regret, but without the noise.
        self.rewards_mean_optimal = np.array(
            [self.rewards_mean[t, self.actions_optimal[t]] for t in range(self.T)]
        )
        self.rewards_mean_chosen = np.array(
            [self.rewards_mean[t, self.actions[t]] for t in range(self.T)]
        )
        self.pseudo_regret = np.cumsum(self.rewards_mean_optimal) - np.cumsum(
            self.rewards_mean_chosen
        )

    def run(self, re_simulate_truth=False):
        """
        Runs the LinUCB algorithm. If re_simulate_truth is True, then the features, parameters and rewards are re-sampled
        before running the algorithm. After running the algorithm, the regret is calculated and assigned as attributes.

        """

        if re_simulate_truth:
            self.simulate()

        self.initialize()

        for t in range(self.T):
            X_t = self.X[t]
            for a in range(self.n_actions):
                self.theta[t, a] = np.linalg.solve(self.A[a], self.b[a])
                self.uncertainty[t, a] = np.sqrt(
                    X_t[a] @ np.linalg.solve(self.A[a], X_t[a])
                )
                self.reward_mean_estimate[t, a] = self.theta[t, a] @ X_t[a]
                self.P[t, a] = (
                    self.reward_mean_estimate[t, a]
                    + self.alpha * self.uncertainty[t, a]
                )
            a_choose = np.argmax(self.P[t])
            self.actions[t] = a_choose
            self.rewards_chosen[t] = self.rewards[t, a_choose]
            self.A[a_choose] += X_t[a_choose].reshape(-1, 1) @ X_t[a_choose].reshape(
                1, -1
            )
            self.b[a_choose] += self.rewards[t, a_choose] * X_t[a_choose]

        self.calc_regret()

    def run_many(self, sims, **kwargs):
        """
        Run the simulation many times so that results can be averaged across simulations. kwargs are passed to the run method.
        """

        cum_payoffs = []
        regrets = []

        for _ in range(sims):
            self.run(**kwargs)
            cum_payoffs.append(np.cumsum(self.rewards_chosen).reshape(1, -1))
            regrets.append(self.regret.reshape(1, -1))

        cum_payoffs = np.concatenate(cum_payoffs, axis=0)
        regrets = np.concatenate(regrets, axis=0)
        return pd.DataFrame(
            dict(
                t=range(cum_payoffs.shape[1]),
                mean_payoff=cum_payoffs.mean(0),
                regret=regrets.mean(0),
            )
        )


class LinUCBSimulatorSimple(LinUCBSimulator):

    """
    This is a simply 1D version of the LinUCB algorithm where parameter values are given as inputs.
    """

    def __init__(
        self,
        T: int,
        error_std_k1: float,
        error_std_k2: float,
        alpha: float,
        theta_k1: float,
        theta_k2: float,
        intercept_k1: float,
        intercept_k2: float,
        lambda_: float = 1,
    ):
        """
        Initialize the LinUCB simulator for the 1D-features and 2-actions case.

        Parameters:
        -----------
        T: int, number of rounds to simulate
        error_std_k1: float, standard deviation of the reward noise for action 1
        error_std_k2: float, standard deviation of the reward noise for action 2
        alpha: float, hyperparameter for the LinUCB algorithm
        theta_k1: float, true coefficient value for action 1
        theta_k2: float, true coefficient value for action 2
        intercept_k1: float, intercept value for action 1
        intercept_k2: float, intercept value for action 2
        lambda_: float, hyperparameter for the LinUCB algorithm
        """
        self.error_std_k1 = error_std_k1
        self.error_std_k2 = error_std_k2
        self.theta_k1 = theta_k1
        self.theta_k2 = theta_k2
        self.intercept_k1 = intercept_k1
        self.intercept_k2 = intercept_k2
        super().__init__(
            T,
            n_actions=2,
            d_features=2,
            error_std=[self.error_std_k1, self.error_std_k2],
            alpha=alpha,
            lambda_=lambda_,
        )

    def simulate_X_and_true_theta(self):
        """
        Generates the context and the true theta parameters
        """

        self.X = np.random.uniform(
            low=X_RANGE[0],
            high=X_RANGE[1],
            size=(self.T, self.n_actions, self.d_features),
        )
        self.X[:, :, 0] = 1.0  # Intercept term
        self.theta_true = np.array(
            [[self.intercept_k1, self.theta_k1], [self.intercept_k2, self.theta_k2]]
        )[np.newaxis, ...]

    def get_X_a_df(self, t, action):
        """
        Returns the context and reward data for a specific time and action.
        """
        return pd.DataFrame(self.X[:t, action, 1], columns=["x"]).assign(
            reward=[
                r if a == action else np.nan
                for r, a in zip(self.rewards_chosen[:t], self.actions[:t])
            ]
        )
