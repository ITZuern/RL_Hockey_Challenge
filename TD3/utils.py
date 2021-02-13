import gym
import numpy as np
import matplotlib.pyplot as plt
import laserhockey.hockey_env as h_env


def plot_learning_curve(x, scores, figure_file):
    running_avg = np.zeros(len(scores))
    for i in range(len(running_avg)):
        running_avg[i] = np.mean(scores[max(0, i-100):(i+1)])
    plt.plot(x, running_avg)
    plt.title('Running average of previous 100 scores')
    plt.savefig(figure_file)


def loadEnv(env_name):
    switch = {
        "hockey_basic_opponent": h_env.HockeyEnv(),
        "hockey_weak_opponent": h_env.HockeyEnv(),
        "hockey_model_opponent": h_env.HockeyEnv(),
        "hockey_train_shoot": h_env.HockeyEnv(mode=h_env.HockeyEnv.TRAIN_SHOOTING),
        "hockey_train_def": h_env.HockeyEnv(mode=h_env.HockeyEnv.TRAIN_DEFENSE)
    }
    env = switch.get(env_name)
    if env:
        return env
    else:
        return gym.make(env_name)


def loadOpponent(env_name):
    switch = {
        "hockey_basic_opponent": h_env.BasicOpponent(weak=False),
        "hockey_weak_opponent": h_env.BasicOpponent(weak=True),
        "hockey_model_opponent": h_env.BasicOpponent(weak=True),
        "hockey_train_shoot": None,
        "hockey_train_def": None
    }
    return switch.get(env_name)


def rewardManipulation(info):
    # Punishment for doing nothing
    reward = -0.005
    # Touch puck reward
    reward += info['reward_touch_puck'] * 30
    # Puck direction reward
    # if info['reward_puck_direction'] > 0:
    #   reward += info['reward_puck_direction'] * 100
    # Winner
    if info['winner'] > 0:
        reward += info['winner'] * 50

    if info['winner'] < 0:
        reward += info['winner'] * 100

    return reward


def countResults(winner, win, lose, draw):
    if winner > 0:
        win += 1
    elif winner < 0:
        lose += 1
    else:
        draw += 1

    return win, lose, draw
