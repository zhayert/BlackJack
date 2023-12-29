from enum import Enum
from random import randint
import numpy as np
from statistics import mean
# from utils.tests import State, states
import random
import matplotlib.pyplot as plt
import seaborn as sns


class State:
    player_sum = 0  # between 12 and 21
    player_usable_ace = False

    dealer_card = 0  # between 2 and 11

    def __init__(self, player_sum, player_usable_ace, dealer_card):
        """Initialize state

        Arguments:
            player_sum {int} -- Player sum
            player_usable_ace {bool} -- Does player have usable ace
            dealer_card {int} -- Dealer card value (11 if ace)
        """
        self.player_sum = player_sum
        self.player_usable_ace = player_usable_ace
        self.dealer_card = dealer_card

    def __eq__(self, other):
        return (self.player_sum, self.player_usable_ace, self.dealer_card) == (
            other.player_sum, other.player_usable_ace, other.dealer_card)

    def __hash__(self):
        return hash((self.player_sum, self.player_usable_ace, self.dealer_card))

    def __ne__(self, other):
        return not (self == other)

    def __repr__(self):
        return f"{self.dealer_card} {self.player_sum} {self.player_usable_ace}"


class Action(Enum):
    HIT = 0
    STAND = 1


class Card:
    """Generate one card and store its number in value field
    """
    value = 0

    def __init__(self):
        number = randint(1, 13)
        if number == 1:
            self.value = 11
        elif number > 10:
            self.value = 10
        else:
            self.value = number


class Dealer:

    def __init__(self):
        self.sum = 0
        self.usable_ace = False
        self.take_card()

    def take_card(self):
        """Take one card

        Returns:
            int -- value of card taken
        """
        c = Card().value
        if c == 11: # when you got a usable_ace
            self.usable_ace = True
        self.sum += c
        return c

    def play_to_end(self):
        """Play while sum < 17

        Returns:
            int -- sum of cards after playing
        """
        while (self.sum < 17):
            self.take_card()
            if self.sum > 21 and self.usable_ace:
                self.usable_ace = False
                self.sum -= 10
        return self.sum


class Player:
    def __init__(self):
        self.sum = 0
        self.usable_ace = False
        while self.sum < 12:
            self.hit()

    def hit(self):
        """Takes one card

        Returns:
            int -- sum after taking card
        """
        c = Card().value
        if c == 11 and self.sum >= 11:
            self.sum += 1
        else:
            if c == 11 and self.sum < 11:
                self.usable_ace = True
            self.sum += c
            if self.sum > 21 and self.usable_ace:
                self.usable_ace = False
                self.sum -= 10
        return self.sum


class Deterministic():
    def __init__(self):
        pass

    def calculate(self, states):
        policy = {}
        for s in states:
            if s.player_usable_ace:
                if s.player_sum <= 17:
                    policy[s] = Action.HIT
                elif s.player_sum == 18 and s.dealer_card >= 9:
                    policy[s] = Action.HIT
                else:
                    policy[s] = Action.STAND
            else:
                if s.player_sum <= 16 and s.dealer_card >= 7:
                    policy[s] = Action.HIT
                elif s.player_sum <= 12 and s.dealer_card <= 3:
                    policy[s] = Action.HIT
                else:
                    policy[s] = Action.STAND
        return policy

    def play_many_times(self, policy, times=30000):
        times_won = 0
        times_draw = 0
        times_lost = 0
        all = 0
        for i in range(0, times):
            result = self.play(policy)
            # if result != 0:
            all += 1
            if result == 1:
                times_won += 1
            elif result == 0:
                times_draw += 1
            else:
                times_lost += 1
        return times_won / all, times_draw / all, times_lost / all

    def play(self, policy, win=1, tie=0, lose=-1):
        player = Player()
        dealer = Dealer()

        state = State(player.sum, player.usable_ace, dealer.sum)
        current_action = policy[state]

        while True:
            if current_action == Action.HIT:
                player.hit()
                if player.sum > 21:
                    return lose
                else:
                    current_state = State(
                        player.sum, player.usable_ace, dealer.sum)
                    current_action = policy[current_state]
            else:
                dealer.play_to_end()
                if dealer.sum > 21 or dealer.sum < player.sum:
                    return win
                elif dealer.sum == player.sum:
                    return tie
                else:
                    return lose

    def print_differences(self, first_policy, second_policy):
        number = 0
        for k, v in first_policy.items():
            if second_policy[k] != v:
                number += 1
                print(k)
                print("FIRST POLICY: ", v)
                print("SECOND POLICY: ", second_policy[k])
                print()

        print("Number of diffrences: ", number)
        return number


class MonteCarlo():
    DELTA = 0.05
    TIMES_TESTED_SAME_STATE = 1000

    def __init__(self, states, epsilon=0.2, improve=True, *_, **__):
        self.IMPROVE = improve
        self.EPSILON = epsilon
        self.policy = {k: {} for k in states}
        self.returns = {(k, a): [] for k in states for a in list(Action)}
        self.Q = {k: {} for k in states}

        # 完成了对状态-动作值函数和策略的初始化，为后续的强化学习算法做好了准备。
        for k in states: # a
            last = 1
            for a in list(Action):
                self.Q[k][a] = 0
                # prob意思就是Action中Hit的概率和STAND的概率加起来等于1
                self.policy[k][a] = random.random() if last == 1 else 1 - last
                last = self.policy[k][a]

    def get_best_action(self, state):
        return max(self.Q[state], key=self.Q[state].get)

    def calculate(self, states, number=10000):
        """Estimate many times

        Keyword Arguments:
            number {int} -- Number of times to estimate (default: {1000000})

        Returns:
            Dictionary -- Estimated policy
        """
        for i in range(0, number):
            self.estimate_one(states)

        if self.IMPROVE:
            for k, v in self.Q.items():
                if abs(v[Action.HIT] - v[Action.STAND]) < self.DELTA:
                    for i in range(0, self.TIMES_TESTED_SAME_STATE):
                        self.estimate_one(states, k)

        return_policy = {}
        for state in self.policy.keys():
            best_action = self.get_best_action(state)
            return_policy[state] = best_action

        return return_policy

    def estimate_one(self, states, starting_state=None):
        episode, reward = self.generate_episode(states)
        for state, action in episode:
            self.returns[(state, action)].append(reward)
            self.Q[state][action] = mean(self.returns[(state, action)])

        for state, _ in episode:
            best_action = self.get_best_action(state)
            for action in list(Action):
                if action == best_action:
                    self.policy[state][action] = 1 - self.EPSILON + (self.EPSILON / len(list(Action)))
                else:
                    self.policy[state][action] = self.EPSILON / len(list(Action))

    def get_action(self, state):
        action_with_probabilities = self.policy[state]
        actions = list(action_with_probabilities.keys())
        probabilities = list(action_with_probabilities.values())
        # choose the larger probability's action
        current_action = np.random.choice(a=actions, size=1, p=probabilities)
        return current_action[0]

    def generate_episode(self, states, starting_state=None):
        """Generate episode using current policy

        Returns:
            list(state, action) -- List of pairs(state, action)
            int -- reward (1, 0 or -1)
        """
        # 创建一个新的玩家（Player）和庄家（Dealer）对象。
        player = Player()
        dealer = Dealer()
        # 根据是否指定了初始状态，选择起始状态。如果没有指定，随机选择一个初始状态。
        current_state = starting_state if starting_state is not None else random.choice(states)
        current_action = self.get_action(current_state)

        player.sum = current_state.player_sum
        player.usable_ace = current_state.player_usable_ace
        dealer.sum = current_state.dealer_card

        episode = [(current_state, current_action)]
        reward = 0

        while True:
            if current_action == Action.HIT:
                player.hit()
                if player.sum > 21:
                    reward = -1
                    break
                else:
                    current_state = State(
                        player.sum, player.usable_ace, dealer.sum)
                    current_action = self.get_action(current_state)
                    episode.append((current_state, current_action))
            elif current_action == Action.STAND:
                dealer.play_to_end()
                if dealer.sum > 21 or dealer.sum < player.sum:
                    reward = 1
                elif dealer.sum == player.sum:
                    reward = 0
                else:
                    reward = -1
                break

        return episode, reward


class QLearning():
    DELTA = 0.1
    TIMES_TESTED_SAME_STATE = 1000

    def __init__(self, states, epsilon=0.2, alpha=0.1, improve=True, *_, **__):
        self.ALPHA = alpha
        self.EPSILON = epsilon
        self.IMPROVE = improve
        self.policy = {k: {} for k in states}
        self.returns = {(k, a): [] for k in states for a in list(Action)}
        self.Q = {k: {} for k in states}

        for k in states:
            last = 1
            for a in list(Action):
                self.Q[k][a] = 0
                self.policy[k][a] = random.random() if last == 1 else 1 - last
                last = self.policy[k][a]

    def calculate(self, states, number=1000000):
        """Estimate many times

        Keyword Arguments:
            number {int} -- Number of times to estimate (default: {1000000})

        Returns:
            Dictionary -- Estimated policy
        """
        for i in range(0, number):
            self.estimate_one(states)

        if self.IMPROVE:
            for k, v in self.Q.items():
                if abs(v[Action.HIT] - v[Action.STAND]) < self.DELTA:
                    for i in range(0, self.TIMES_TESTED_SAME_STATE):
                        self.estimate_one(states, k)

        return_policy = {}
        for state in self.policy.keys():
            best_action = self.get_best_action(state)
            return_policy[state] = best_action

        return return_policy

    def get_best_action(self, state):
        return max(self.Q[state], key=self.Q[state].get)

    def estimate_one(self, states, starting_state=None):
        episode, reward = self.generate_episode(states)
        for i in range(0, len(episode) - 1):
            current_state = episode[i][0]
            current_action = episode[i][1]

            next_state = episode[i + 1][0]
            next_action = episode[i + 1][1]

            diff = self.Q[next_state][self.get_best_action(next_state)] - self.Q[current_state][current_action]

            self.Q[current_state][current_action] += self.ALPHA * diff

        self.Q[episode[len(episode) - 1][0]][episode[len(episode) - 1][1]] += self.ALPHA * (
                reward - self.Q[episode[len(episode) - 1][0]][episode[len(episode) - 1][1]])

        for state, _ in episode:
            best_action = self.get_best_action(state)
            for action in list(Action):
                if action == best_action:
                    self.policy[state][action] = 1 - self.EPSILON + (self.EPSILON / len(list(Action)))
                else:
                    self.policy[state][action] = self.EPSILON / len(list(Action))

    def get_action(self, state):
        action_with_probabilities = self.policy[state]
        actions = list(action_with_probabilities.keys())
        probabilities = list(action_with_probabilities.values())

        current_action = np.random.choice(a=actions, size=1, p=probabilities)
        return current_action[0]

    def generate_episode(self, states, starting_state=None):
        """Generate episode using current policy

        Returns:
            list(state, action) -- List of pairs(state, action)
            int -- reward (1 or 0)
        """
        player = Player()
        dealer = Dealer()

        current_state = starting_state if starting_state is not None else random.choice(states)
        current_action = self.get_action(current_state)

        player.sum = current_state.player_sum
        player.usable_ace = current_state.player_usable_ace
        dealer.sum = current_state.dealer_card

        episode = [(current_state, current_action)]
        reward = 0

        while True:
            if current_action == Action.HIT:
                player.hit()
                if player.sum > 21:
                    reward = -1
                    break
                else:
                    current_state = State(
                        player.sum, player.usable_ace, dealer.sum)
                    current_action = self.get_action(current_state)
                    episode.append((current_state, current_action))
            else:
                dealer.play_to_end()
                if dealer.sum > 21 or dealer.sum < player.sum:
                    reward = 1
                elif dealer.sum == player.sum:
                    reward = 0
                else:
                    reward = -1
                break

        return episode, reward


class Sarsa():
    DELTA = 0.1
    TIMES_TESTED_SAME_STATE = 1000

    def __init__(self, states, epsilon=0.2, alpha=0.1, improve=True, *_, **__):
        self.ALPHA = alpha
        self.EPSILON = epsilon
        self.IMPROVE = improve
        self.policy = {k: {} for k in states}
        self.returns = {(k, a): [] for k in states for a in list(Action)}
        self.Q = {k: {} for k in states}

        for k in states:
            last = 1
            for a in list(Action):
                self.Q[k][a] = 0
                self.policy[k][a] = random.random() if last == 1 else 1 - last
                last = self.policy[k][a]

    def calculate(self, states, number=1000000):
        """Estimate many times

        Keyword Arguments:
            number {int} -- Number of times to estimate (default: {1000000})

        Returns:
            Dictionary -- Estimated policy
        """
        for i in range(0, number):
            self.estimate_one(states)

        if self.IMPROVE:
            for k, v in self.Q.items():
                if abs(v[Action.HIT] - v[Action.STAND]) < self.DELTA:
                    for i in range(0, self.TIMES_TESTED_SAME_STATE):
                        self.estimate_one(states, k)

        return_policy = {}
        for state in self.policy.keys():
            best_action = self.get_best_action(state)
            return_policy[state] = best_action

        return return_policy

    def get_best_action(self, state):
        return max(self.Q[state], key=self.Q[state].get)

    def estimate_one(self, states, starting_state=None):
        episode, reward = self.generate_episode(states)
        for i in range(0, len(episode) - 1):
            current_state = episode[i][0]
            current_action = episode[i][1]

            next_state = episode[i + 1][0]
            next_action = episode[i + 1][1]

            diff = self.Q[next_state][next_action] - self.Q[current_state][current_action]

            self.Q[current_state][current_action] += self.ALPHA * diff

        self.Q[episode[len(episode) - 1][0]][episode[len(episode) - 1][1]] += self.ALPHA * (
                reward - self.Q[episode[len(episode) - 1][0]][episode[len(episode) - 1][1]])

        for state, _ in episode:
            best_action = self.get_best_action(state)
            for action in list(Action):
                if action == best_action:
                    self.policy[state][action] = 1 - self.EPSILON + (self.EPSILON / len(list(Action)))
                else:
                    self.policy[state][action] = self.EPSILON / len(list(Action))

    def get_action(self, state):
        action_with_probabilities = self.policy[state]
        actions = list(action_with_probabilities.keys())
        probabilities = list(action_with_probabilities.values())

        current_action = np.random.choice(a=actions, size=1, p=probabilities)
        return current_action[0]

    def generate_episode(self, states, starting_state=None):
        """Generate episode using current policy

        Returns:
            list(state, action) -- List of pairs(state, action)
            int -- reward (1 or 0)
        """
        player = Player()
        dealer = Dealer()

        current_state = starting_state if starting_state is not None else random.choice(states)
        current_action = self.get_action(current_state)

        player.sum = current_state.player_sum
        player.usable_ace = current_state.player_usable_ace
        dealer.sum = current_state.dealer_card

        episode = [(current_state, current_action)]
        reward = 0

        while True:
            if current_action == Action.HIT:
                player.hit()
                if player.sum > 21:
                    reward = -1
                    break
                else:
                    current_state = State(
                        player.sum, player.usable_ace, dealer.sum)
                    current_action = self.get_action(current_state)
                    episode.append((current_state, current_action))
            else:
                dealer.play_to_end()
                if dealer.sum > 21 or dealer.sum < player.sum:
                    reward = 1
                elif dealer.sum == player.sum:
                    reward = 0
                else:
                    reward = -1
                break

        return episode, reward

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


if __name__ == '__main__':
    print(list(Action))