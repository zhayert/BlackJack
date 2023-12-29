import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

class BlackjackPlotter:
    def __init__(self, states, algorithm, epsilon, alpha, improve, number):
        self.states = states
        self.algorithm = algorithm
        self.epsilon = epsilon
        self.alpha = alpha
        self.improve = improve
        self.number = number
        self.deterministic_agent = None
        self.agent = None

    def run_algorithm(self):
        algos = {"mcc": MonteCarlo,
                 "ql": QLearning,
                 "sarsa": Sarsa}

        algo = None
        for name, cls in algos.items():
            if name.lower().startswith(self.algorithm.lower()):
                algo = cls

        self.deterministic_agent = Deterministic()
        self.agent = algo(states=self.states, epsilon=self.epsilon, alpha=self.alpha, improve=self.improve)

        policy = self.agent.calculate(self.states, self.number)
        return policy

    def plot_results(self, policy):
        state_action_values = np.zeros((len(self.states), 2, len(Action)))

        for i, state in enumerate(self.states):
            for j, action in enumerate(list(Action)):
                state_action_values[i, 0 if state.player_usable_ace else 1, j] = self.agent.Q[state][action]

        state_value_no_usable_ace = np.max(state_action_values[:, 1, :], axis=-1)
        state_value_usable_ace = np.max(state_action_values[:, 0, :], axis=-1)

        action_no_usable_ace = np.argmax(state_action_values[:, 1, :], axis=-1)
        action_usable_ace = np.argmax(state_action_values[:, 0, :], axis=-1)

        images = [action_usable_ace,
                  state_value_usable_ace,
                  action_no_usable_ace,
                  state_value_no_usable_ace]

        titles = ['Optimal policy with usable Ace',
                  'Optimal value with usable Ace',
                  'Optimal policy without usable Ace',
                  'Optimal value without usable Ace']

        _, axes = plt.subplots(2, 2, figsize=(15, 10))
        plt.subplots_adjust(wspace=0.1, hspace=0.2)
        axes = axes.flatten()

        for image, title, axis in zip(images, titles, axes):
            sns.heatmap(np.flipud(image), cmap="YlGnBu", ax=axis, xticklabels=range(1, 11),
                        yticklabels=list(reversed(range(12, 22))))
            axis.set_ylabel('Player Sum', fontsize=12)
            axis.set_xlabel('Dealer Showing', fontsize=12)
            axis.set_title(title, fontsize=12)

        plt.show()

# Example usage:
states = []  # Populate with your actual states
algorithm = "mcc"
epsilon = 0.2
alpha = 0.02
improve = True
number = 10000

plotter = BlackjackPlotter(states, algorithm, epsilon, alpha, improve, number)
policy = plotter.run_algorithm()
plotter.plot_results(policy)