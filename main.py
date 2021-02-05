import gym
import numpy as np
import optparse
from agent import Agent
import laserhockey.hockey_env as h_env
from utils import plot_learning_curve

if __name__ == '__main__':
    # LunarLanderContinuous-v2
    # Pendulum-v0
    # hockey_shoot

    optParser = optparse.OptionParser()
    optParser.add_option('-e', '--env', action='store', type='string',
                         dest='env_name', default="hockey_shoot",
                         help='Environment (default %default)')
    optParser.add_option('-t', '--train', action='store_true',
                         dest='train', default=False)
    optParser.add_option('-p', '--path', action='store',
                         dest='path', default="", type='string')
    optParser.add_option('-n', '--n_games', action='store',
                         dest='n_games', default=5, type='int')
    optParser.add_option('-r', '--render', action='store_true',
                         dest='render', default=False)
    optParser.add_option('-s', '--seed', action='store',  type='int',
                         dest='seed', default=0,
                         help='random seed (default %default)')

    opts, args = optParser.parse_args()
    env_name = opts.env_name
    train = opts.train
    load_path = "models/"+opts.path
    explore = train
    render = opts.render
    scores = []
    n_games = opts.n_games
    noise = 0.12
    seed = opts.seed

    if env_name == "hockey_shoot":
        # h_env.HockeyEnv()
        env = h_env.HockeyEnv(mode=h_env.HockeyEnv.TRAIN_SHOOTING)
    else:
        env = gym.make(env_name)

    # only use half of the action dim because we only want to control one player
    action_dim = int(env.action_space.shape[0])  # /2
    if env_name == "hockey_shoot":
        action_dim = int(env.action_space.shape[0]/2)

    agent = Agent(alpha=0.0001, beta=0.001,
                  input_dims=env.observation_space.shape, tau=0.001,
                  batch_size=64, fc1_dims=400, fc2_dims=300,
                  n_actions=action_dim)

    if opts.path != "":
        agent.load(load_path)

    print("TRAIN: ", train)
    print("EXPLORE: ", explore)
    print("PATH: ", opts.path)
    print("NOISE: ", noise)

    filename = env_name + str(agent.alpha) + '_beta_' + \
        str(agent.beta) + '_' + str(n_games) + '_games'
    figure_file = 'plots/' + filename + '.png'

    best_score = env.reward_range[0]
    score_history = []
    for i in range(n_games):
        observation = env.reset()
        done = False
        score = 0
        agent.noise.reset()
        while not done:
            if env_name == "hockey_shoot":
                action_p1 = agent.act(observation, explore)
                ### FREEZE OPPONENT ###
                action_p2 = [0, 0, 0, 0]
                #action_p1 = [0,0,0,0]
                action = np.hstack([action_p1, action_p2])
            else:
                action = agent.act(observation, explore)

            observation_, reward, done, info = env.step(action)

            reward * 2000
            ##### REDO ####
            if reward < 0:
                reward *= -1
            if reward == 0.0:
                reward = -10
            ##### REDO ####

            if env_name == "hockey_shoot":
                agent.remember(observation, action_p1,
                               reward, observation_, done)
            else:
                agent.remember(observation, action, reward, observation_, done)

            if train:
                agent.learn()

            score += reward
            observation = observation_

            if render:
                env.render()

        score_history.append(score)
        avg_score = np.mean(score_history[-100:])

        print('episode ', i, 'score %.1f' % score,
              'average score %.1f' % avg_score)
    x = [i+1 for i in range(n_games)]

    if train:
        agent.save(env_name)
        plot_learning_curve(x, score_history, figure_file)
