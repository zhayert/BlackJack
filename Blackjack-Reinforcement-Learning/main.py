from blackjack import Deterministic, MonteCarlo, QLearning, Sarsa, Player, Dealer, Action, State, BlackjackPlotter
import time

algorithms = {"mcc": MonteCarlo,
         "ql": QLearning,
         "sarsa": Sarsa}

number = 10000
epsilon = 0.2
alpha = 0.02
improve = True
model = None
states = []
algorithm = "mcc"

for player_usable_ace in [False, True]:
    for player_sum in range(12, 22):
        for dealer_card in range(2, 12):
            states.append(State(player_sum, player_usable_ace, dealer_card))

for name, cls in algorithms.items():
    if name == algorithm:
        model = cls

deterministic_agent = Deterministic()
agent = model(states=states, epsilon=epsilon, alpha=alpha, improve=improve)

start_time = time.time()
policy = agent.calculate(states, number)
elapsed_time = print("TIME: ", time.time() - start_time)

deterministic_agent.print_differences(deterministic_agent.calculate(states=states), policy)

print(deterministic_agent.play_many_times(policy, times=1000000))
print(policy)
