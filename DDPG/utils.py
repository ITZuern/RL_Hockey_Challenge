import gym
import numpy as np
import matplotlib.pyplot as plt
import laserhockey.hockey_env as h_env
from agent import Agent


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
        "hockey_train_shoot": h_env.HockeyEnv(mode=h_env.HockeyEnv.TRAIN_SHOOTING),
        "hockey_train_def": h_env.HockeyEnv(mode=h_env.HockeyEnv.TRAIN_DEFENSE)
    }
    env = switch.get(env_name)
    if env:
        return env
    else:
        return h_env.HockeyEnv()


def loadOpponent(env_name):
    switch = {
        "hockey_basic_opponent": h_env.BasicOpponent(weak=False),
        "hockey_weak_opponent": h_env.BasicOpponent(weak=True),
        "hockey_train_shoot": None,
        "hockey_train_def": None
    }
    opponent = switch.get(env_name)
    # if env_name was non of the options above, it is the path for a self trained model
    if env_name not in ["hockey_basic_opponent",  "hockey_weak_opponent", "hockey_train_shoot", "hockey_train_def"]:
        # init agent
        env = h_env.HockeyEnv()
        action_dim = int(env.action_space.shape[0]/2)
        opponent = Agent(alpha=0.0001, beta=0.001,
                         input_dims=env.observation_space.shape, tau=0.001,
                         batch_size=64, fc1_dims=400, fc2_dims=300,
                         n_actions=action_dim, device="cpu")
        # load pretrained agent
        opponent.load("models/"+env_name)
    return opponent


def rewardManipulation(info, iteration, done):
    reward = 0
    if iteration < 500:
        if done:
            if info['winner'] == 0:
                reward -= 200

        reward += info['reward_touch_puck'] * 30
        reward -= 0.005

    elif iteration < 1000:
        if done:
            if info['winner'] == 0:
                reward -= 50

            if info['winner'] == 1:
                reward += 500

            if info['winner'] == -1:
                reward -= 200

        reward += info['reward_touch_puck'] * 30

    elif iteration < 2000:
        if done:
            if info['winner'] == 0:
                reward -= 200

            if info['winner'] == 1:
                reward += 700

            if info['winner'] == -1:
                reward -= 500

    else:
        if done:
            if info['winner'] == 0:
                reward -= 700

            if info['winner'] == 1:
                reward += 700

            if info['winner'] == -1:
                reward -= 700

    return reward


def countResults(winner, win, lose, draw):
    if winner > 0:
        win += 1
    elif winner < 0:
        lose += 1
    else:
        draw += 1

    return win, lose, draw
