import gym
import numpy as np
import optparse
from agent import Agent
import laserhockey.hockey_env as h_env
from utils import *
from datetime import datetime

if __name__ == '__main__':
    # LunarLanderContinuous-v2
    # Pendulum-v0
    # hockey_shoot

    optParser = optparse.OptionParser()
    optParser.add_option('-e', '--env', action='store', type='string',
                         dest='env_name', default="LunarLanderContinuous-v2",
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
    scores = []
    n_games = opts.n_games
    seed = opts.seed
    opponent = opts.opponent

    env = loadEnv(env_name)

    # only use half of the action dim because we only want to control one player
    action_dim = int(env.action_space.shape[0])  # /2
    print(action_dim)
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

    score_history = []
    wins = 0
    loses = 0
    draws = 0
    player2 = h_env.BasicOpponent(weak=False)
    for i in range(n_games):
        observation = env.reset()
        #obs_agent2 = env.obs_agent_two()
        done = False
        score = 0
        # agent.noise.reset()
        while not done:
            if env_name.startswith('hockey'):
                action_p1 = agent.act(observation, explore)
                ### FREEZE OPPONENT ###
                #action_p2 = player2.act(obs_agent2)
                # action_p2 = [0, 0, 0, 0]
                #action = np.hstack([action_p1, action_p2])
            else:
                action = agent.act(observation, explore)

            observation_, reward, done, info = env.step(action)

            #reward_puck_direction = info['reward_puck_direction']
            #reward_touch_puck = info['reward_touch_puck']
            # Buggy
            #reward_closeness_to_puck = info['reward_closeness_to_puck']
            #winner = info['winner']

            ###Reward manipulation###
            # Punishment for doing nothing
            #reward = -0.005
            # Touch puck reward
            #reward += info['reward_touch_puck'] * 30
            # Puck direction reward
            # if info['reward_puck_direction'] > 0:
            #   reward += info['reward_puck_direction'] * 100
            # Winner
            # if info['winner'] > 0:
            #    wins += 1
            #    reward += info['winner'] * 50

            # if info['winner'] < 0:
            #    loses += 1
            #    reward += info['winner'] * 100
            ###End Reward manipulation###

            if env_name.startswith('hockey'):
                agent.remember(observation, action_p1,
                               reward, observation_, done)
            else:
                agent.remember(observation, action, reward, observation_, done)

            if train:
                agent.learn()

            score += reward
            observation = observation_
            #obs_agent2 = env.obs_agent_two()

            if render:
                env.render()

        # if info['winner'] == 0:
        #    draws += 1

        score_history.append(score)
        avg_score = np.mean(score_history[-50:])

        print('episode ', i, 'score %.1f' % score,
              'average score %.1f' % avg_score)
        print('Winrate in percent: ', (wins / (i+1) * 100))
        print('Drawrate in percent: ', (draws / (i+1) * 100))
        print('Loserate in percent: ', (loses / (i+1) * 100))
    x = [i+1 for i in range(n_games)]

    if train:
        agent.save(env_name)
        filename = env_name + str(datetime.now().strftime("-%m%d%Y%H%M%S"))
        figure_file = 'plots/' + filename + '.png'
        plot_learning_curve(x, score_history, figure_file)


def play(method):
    switch = {
        "basic": playBasic(agent, env, train, render, n_games, opponent),
        "mix_v1": h_env.HockeyEnv(mode=h_env.HockeyEnv.TRAIN_SHOOTING),
        "mix_v2": h_env.HockeyEnv(mode=h_env.HockeyEnv.TRAIN_DEFENSE),
    }


def mix_v1():
    print("Mix v1 Mode")


def playBasic(agent, env, train, render, n_games, opponent):
    if opponent is None:
        standardSetup(agent, env, train, render, n_games)

    elif opponent == 'basic':
        basicOpponentSetup(agent, env, train, render, n_games, False)

    elif opponent == 'weak':
        basicOpponentSetup(agent, env, train, render, n_games, True)

    else:
        otherModelSetup(agent, env, train, render, n_games, opponent)


def standardSetup(agent, env, train, render, n_games):
    print("Standard Setup")
    score_history = []
    for i in range(n_games):
        observation = env.reset()
        done = False
        score = 0
        agent.noise.reset()
        while not done:
            action = agent.act(observation, explore)
            observation_, reward, done, _ = env.step(action)
            agent.remember(observation, action, reward, observation_, done)

            if train:
                agent.learn()

            score += reward
            observation = observation_

            if render:
                env.render()

        score_history.append(score)
        avg_score = np.mean(score_history[-50:])

        print('episode ', i, 'score %.1f' % score,
              'average score %.1f' % avg_score)
        print('Winrate in percent: ', (wins / (i+1) * 100))
        print('Drawrate in percent: ', (draws / (i+1) * 100))
        print('Loserate in percent: ', (loses / (i+1) * 100))

    return score_history


def basicOpponentSetup(agent, env, train, render, n_games, weak):
    print("BasicOpponent Setup, weak=", str(weak))
    score_history = []
    wins, loses, draws = 0, 0, 0
    player2 = h_env.BasicOpponent(weak=weak)
    for i in range(n_games):
        observation = env.reset()
        obs_agent2 = env.obs_agent_two()
        done = False
        score = 0
        agent.noise.reset()
        while not done:
            action_p1 = agent.act(observation, explore)
            action_p2 = player2.act(obs_agent2)
            action = np.hstack([action_p1, action_p2])

            observation_, _, done, info = env.step(action)

            reward = rewardManipulation(info)

            wins, loses, draws = countResults(
                info['winner'], wins, loses, draws)

            agent.remember(observation, action_p1,
                           reward, observation_, done)
            if train:
                agent.learn()

            score += reward
            observation = observation_
            obs_agent2 = env.obs_agent_two()

            if render:
                env.render()

        score_history.append(score)
        avg_score = np.mean(score_history[-50:])

        print('episode ', i, 'score %.1f' % score,
              'average score %.1f' % avg_score)
        print('Winrate in percent: ', (wins / (i+1) * 100))
        print('Drawrate in percent: ', (draws / (i+1) * 100))
        print('Loserate in percent: ', (loses / (i+1) * 100))

    return score_history


def otherModelSetup(agent, env, train, render, n_games, model_path):
    print("Play against another model, model=", model_path)
    agent2 = Agent(alpha=0.0001, beta=0.001,
                   input_dims=env.observation_space.shape, tau=0.001,
                   batch_size=64, fc1_dims=400, fc2_dims=300,
                   n_actions=action_dim)
    agent2.load(model_path)
    score_history = []
    wins, loses, draws = 0, 0, 0
    for i in range(n_games):
        observation = env.reset()
        obs_agent2 = env.obs_agent_two()
        done = False
        score = 0
        agent.noise.reset()
        while not done:
            action_p1 = agent.act(observation, explore)
            action_p2 = agent2.act(obs_agent2, False)
            action = np.hstack([action_p1, action_p2])

            observation_, _, done, info = env.step(action)

            reward = rewardManipulation(info)

            wins, loses, draws = countResults(
                info['winner'], wins, loses, draws)

            agent.remember(observation, action_p1,
                           reward, observation_, done)
            if train:
                agent.learn()

            score += reward
            observation = observation_
            obs_agent2 = env.obs_agent_two()

            if render:
                env.render()

        score_history.append(score)
        avg_score = np.mean(score_history[-50:])

        print('episode ', i, 'score %.1f' % score,
              'average score %.1f' % avg_score)
        print('Winrate in percent: ', (wins / (i+1) * 100))
        print('Drawrate in percent: ', (draws / (i+1) * 100))
        print('Loserate in percent: ', (loses / (i+1) * 100))

    return score_history


def freezeSetup(agent, env, train, render, n_games):
    print("Freeze Setup")
    score_history = []
    wins, loses, draws = 0, 0, 0
    for i in range(n_games):
        observation = env.reset()
        done = False
        score = 0
        agent.noise.reset()
        while not done:
            action_p1 = agent.act(observation, explore)
            action = np.hstack([action_p1, [0, 0, 0, 0]])

            observation_, _, done, info = env.step(action)

            reward = rewardManipulation(info)

            wins, loses, draws = countResults(
                info['winner'], wins, loses, draws)

            agent.remember(observation, action_p1,
                           reward, observation_, done)
            if train:
                agent.learn()

            score += reward
            observation = observation_

            if render:
                env.render()

        score_history.append(score)
        avg_score = np.mean(score_history[-50:])

        print('episode ', i, 'score %.1f' % score,
              'average score %.1f' % avg_score)
        print('Winrate in percent: ', (wins / (i+1) * 100))
        print('Drawrate in percent: ', (draws / (i+1) * 100))
        print('Loserate in percent: ', (loses / (i+1) * 100))

    return score_history
