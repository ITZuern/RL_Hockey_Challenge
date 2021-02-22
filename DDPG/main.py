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

    # Parse options
    opts = parseOptions(optparse.OptionParser())
    env_name = opts.env_name
    train = opts.train
    load_path = "models/"+opts.path
    explore = opts.xplore
    render = opts.render
    n_games = opts.n_games
    device = opts.device
    trainPlan = opts.trainplan
    games_per_env = opts.games_per_env
    turnament = opts.turnament
    save_opponent = opts.save_opponent

    env = loadEnv(env_name)

    # only use half of the action dim because we only want to control one player
    action_dim = int(env.action_space.shape[0])
    if env_name.startswith('hockey'):
        action_dim = int(env.action_space.shape[0]/2)

    # init agent
    agent = Agent(alpha=0.0001, beta=0.001,
                  input_dims=env.observation_space.shape, tau=0.001,
                  batch_size=64, fc1_dims=400, fc2_dims=300,
                  n_actions=action_dim, device=device)

    # if a path for a pretrained agent was given, load this agent
    if opts.path != "":
        agent.load(load_path)

    print("TRAIN: ", train)
    print("EXPLORE: ", explore)
    print("PATH: ", opts.path)
    print("DEVICE: ", device)
    print("TRAIN PLAN: ", trainPlan)

    # Game mode Hockey
    if env_name.startswith('hockey'):
        # build list of training scenarios
        # (shoot, defend, weak opponent, strong opponent or own network)
        env_names = buildTrainPlan(trainPlan)
        # run several games and switch env after certain amount of games
        score_history = playHockey(
            agent, env_names, n_games, explore, train, render, games_per_env, turnament, save_opponent)

    # Save the agent and make plot of reward curve
    if train:
        x = [i+1 for i in range(n_games)]
        agent.save(env_name)
        filename = env_name + str(datetime.now().strftime("-%m%d%Y%H%M%S"))
        figure_file = 'plots/' + filename + '.png'
        plot_learning_curve(x, score_history, figure_file)


def parseOptions(optParser):
    optParser.add_option('-e', '--env', action='store', type='string',
                         dest='env_name', default="hockey_train_shoot",
                         help='Environment (default %default)')
    optParser.add_option('-t', '--train', action='store_true',
                         dest='train', default=False)
    optParser.add_option('-p', '--path', action='store',
                         dest='path', default="", type='string')
    optParser.add_option('-n', '--n_games', action='store',
                         dest='n_games', default=100, type='int')
    optParser.add_option('-r', '--render', action='store_true',
                         dest='render', default=False)
    optParser.add_option('-s', '--seed', action='store', type='int',
                         dest='seed', default=0,
                         help='random seed (default %default)')
    optParser.add_option('-x', '--xplore', action='store_true',
                         dest='xplore', default=False)
    optParser.add_option('-d', '--device', action='store', type='string',
                         dest='device', default='cuda')
    optParser.add_option('-v', '--trainplan', action='store', type='string',
                         dest='trainplan', default='shoot')
    optParser.add_option('-w', '--games_per_env', action='store',
                         dest='games_per_env', default=10, type='int')
    optParser.add_option('-u', '--turnament', action='store_true',
                         dest='turnament', default=False)
    optParser.add_option('-o', '--save_opponent', action='store',
                         dest='save_opponent', default=20, type='int')
    opts, args = optParser.parse_args()
    return opts


def getOpponentAction(env_name, env, opponent):
    if env_name == 'hockey_train_shoot' or env_name == 'hockey_train_def':
        # For shoot and defend training, the opponent is fixed
        return [0, 0, 0, 0]
    elif env_name == 'hockey_basic_opponent' or env_name == 'hockey_weak_opponent':
        # for basic opponent, get action based on obs_agent two
        return opponent.act(env.obs_agent_two())
    else:
        # for own network as opponent, get
        return opponent.act(env.obs_agent_two(), False)


def playHockey(agent, env_names, n_games, explore, train, render, switch=10, turnament=False, save_opponent=20):
    score_history = []
    j = 0
    result_counter = 0
    env = None
    opponent_idx = 0

    # run n games
    for i in range(n_games):

        # if switch training environment after several games
        if i % switch == 0:
            # do not change if only one train env is selected
            if not (env and len(env_names) == 1):
                # reset win, loses and draws counter
                wins, loses, draws = 0, 0, 0
                # switch the training environment
                env_name = env_names[j]
                print('\n \n Switch opponent to: ', env_name)
                if j == len(env_names) - 1:
                    j = 0
                else:
                    j += 1
                result_counter = 0
                opponent = loadOpponent(env_name)
                # close previous environment
                if(env):
                    env.close()
                # load new environment
                env = loadEnv(env_name)

        # rollout of one game
        observation = env.reset()
        done = False
        score = 0
        while not done:
            # get action from agent and opponent
            action_p1 = agent.act(observation, explore)
            action_p2 = getOpponentAction(env_name, env, opponent)
            action = np.hstack([action_p1, action_p2])
            # perform actions and get observations
            observation_, _, done, info = env.step(action)
            # perform training
            # optional reward manipulation for training
            reward = rewardManipulation(info)
            if train:
                # fill buffer of agent for training
                agent.remember(observation, action_p1,
                               reward, observation_, done)
                agent.learn()
            score += reward
            observation = observation_
            # render the game if in render mode
            if render:
                env.render()

        # if turnament mode save new opponent after some games
        if turnament and i % save_opponent == 0 and i != 0:
            print("Add opponent ", opponent_idx+1)
            name = "op_"+str(opponent_idx+1)
            agent.save(name, timestamp=False)
            if name not in env_names:
                env_names.append(name)
            opponent_idx += 1
            # after ten saved opponents the oldest one is overwritten
            if opponent_idx >= 10:
                opponent_idx = 0

        # keep track of wins/loses/draws
        result_counter += 1
        wins, loses, draws = countResults(
            info['winner'], wins, loses, draws)
        # keep track of score(rewards)
        score_history.append(score)
        avg_score = np.mean(score_history[-50:])

        # console output
        print('episode ', i, 'score %.1f' % score,
              'average score %.1f' % avg_score)
        print('Winrate in percent: ', (wins / (result_counter) * 100))
        print('Drawrate in percent: ', (draws / (result_counter) * 100))
        print('Loserate in percent: ', (loses / (result_counter) * 100))

    return score_history


def buildTrainPlan(trainPlan):
    switch = {
        "shoot": ['hockey_train_shoot'],
        "def": ['hockey_train_def'],
        "weak": ['hockey_weak_opponent'],
        "strong": ['hockey_basic_opponent'],
        "static": ['hockey_train_shoot', 'hockey_train_def'],
        "basic": ['hockey_weak_opponent', 'hockey_basic_opponent'],
        "full": ['hockey_train_shoot', 'hockey_train_def', 'hockey_weak_opponent', 'hockey_basic_opponent'],
        "friedo": ['wuetender_walter'],
        "v1": ['hockey_train_shoot', 'hockey_train_def', 'hockey_weak_opponent', 'hockey_basic_opponent', 'test'],
        "v2": ['hockey_weak_opponent', 'hockey_basic_opponent', 'wuetender_walter_v2']
    }
    return switch.get(trainPlan)


main()
