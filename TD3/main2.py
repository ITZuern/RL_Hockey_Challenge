import gym
import numpy as np
import optparse
from agent import Agent
import laserhockey.hockey_env as h_env
from utils import *
from datetime import datetime


def main():
    # LunarLanderContinuous-v2
    # Pendulum-v0
    # hockey_shoot

    optParser = optparse.OptionParser()
    optParser.add_option('-e', '--env', action='store', type='string',
                         dest='env_name', default="hockey_train_shoot",
                         help='Environment (default %default)')
    optParser.add_option('-t', '--train', action='store_true',
                         dest='train', default=False)
    optParser.add_option('-p', '--path', action='store',
                         dest='path', default="", type='string')
    optParser.add_option('-n', '--n_games', action='store',
                         dest='n_games', default=5, type='int')
    optParser.add_option('-r', '--render', action='store_true',
                         dest='render', default=False)
    optParser.add_option('-s', '--seed', action='store', type='int',
                         dest='seed', default=0,
                         help='random seed (default %default)')
    optParser.add_option('-x', '--xplore', action='store_true',
                         dest='xplore', default=False)
    optParser.add_option('-o', '--opponent', action='store', type='string',
                         dest='opponent', default='none')

    opts, args = optParser.parse_args()
    env_name = opts.env_name
    train = opts.train
    load_path = "models/"+opts.path
    explore = opts.xplore
    render = opts.render
    n_games = opts.n_games

    env = loadEnv(env_name)

    # only use half of the action dim because we only want to control one player
    action_dim = int(env.action_space.shape[0])  # /2
    if env_name.startswith('hockey'):
        action_dim = int(env.action_space.shape[0]/2)

    agent = Agent(alpha=0.001, beta=0.001,
                  input_dims=env.observation_space.shape, tau=0.005,
                  env=env, batch_size=100, layer1_size=400, layer2_size=300,
                  n_actions=action_dim)

    if opts.path != "":
        agent.load(load_path)

    print("TRAIN: ", train)
    print("EXPLORE: ", explore)
    print("PATH: ", opts.path)

    if env_name.startswith('hockey'):
        env_names = ['hockey_train_shoot', 'hockey_weak_opponent']
        score_history = playHockey(
            agent, env_names, n_games, explore, train, render, 10)

    x = [i+1 for i in range(n_games)]

    if train:
        agent.save(env_name)
        filename = env_name + str(datetime.now().strftime("-%m%d%Y%H%M%S"))
        figure_file = 'plots/' + filename + '.png'
        plot_learning_curve(x, score_history, figure_file)


def getOpponentAction(env_name, env, opponent):
    if env_name == 'hockey_train_shoot' or env_name == 'hockey_train_def':
        return [0, 0, 0, 0]
    elif env_name == 'hockey_basic_opponent' or env_name == 'hockey_weak_opponent':
        return opponent.act(env.obs_agent_two())
    else:
        return opponent.act(env.obs_agent_two(), False)


def playHockey(agent, env_names, n_games, explore, train, render, switch=100):
    score_history = []
    j = 0
    result_counter = 0
    for i in range(n_games):
        if i % switch == 0:
            wins, loses, draws = 0, 0, 0
            env_name = env_names[j]
            print('\n \n Switch opponent to: ', env_name)
            if j == len(env_names) - 1:
                j = 0
            else:
                j += 1

            result_counter = 0
            opponent = loadOpponent(env_name)
            env = loadEnv(env_name)

        observation = env.reset()
        done = False
        score = 0
        while not done:
            action_p1 = agent.act(observation, explore)
            action_p2 = getOpponentAction(env_name, env, opponent)
            action = np.hstack([action_p1, action_p2])

            observation_, _, done, info = env.step(action)

            reward = rewardManipulation(info)

            agent.remember(observation, action_p1,
                           reward, observation_, done)

            if train:
                agent.learn()

            score += reward
            observation = observation_

            if render:
                env.render()

        result_counter += 1
        wins, loses, draws = countResults(
            info['winner'], wins, loses, draws)

        score_history.append(score)
        avg_score = np.mean(score_history[-50:])

        print('episode ', i, 'score %.1f' % score,
              'average score %.1f' % avg_score)
        print('Winrate in percent: ', (wins / (result_counter) * 100))
        print('Drawrate in percent: ', (draws / (result_counter) * 100))
        print('Loserate in percent: ', (loses / (result_counter) * 100))

    return score_history


main()
