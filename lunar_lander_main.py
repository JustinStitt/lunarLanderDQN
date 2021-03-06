import gym
from lunar_landerDQN import Agent
import numpy as np
import torch as T
import sys, os


if __name__ == '__main__':
    env = gym.make('LunarLander-v2')
    brain = Agent(gamma = 0.99, epsilon = 1.0, batch_size = 64, n_actions = 4,
                    input_dims = [8], lr = 0.003)

    if os.path.isfile('trained_model.pt'):
        brain.load_checkpoint()

    scores = []
    eps_history = []
    n_games = 500
    score = 0

    for i in range(n_games):
        if i % 10 == 0 and i > 0:
            avg_score = np.mean(scores)
            print('episode: ', i, 'score ', score,
                        'average score %.3f' % avg_score,
                        'epsilon %.3f' % brain.epsilon)
        else:
            print('epsiode ', i, 'score ', score)
        score = 0
        eps_history.append(brain.epsilon)
        observation = env.reset()
        done = False
        while not done:
            env.render()#to draw

            action = brain.choose_action(observation)
            observation_, reward, done, info = env.step(action)
            score += reward
            brain.store_transition(observation, action, reward, observation_, done)
            brain.learn()
            observation = observation_
    brain.save_checkpoint()
        #model.save_state_dict('mytraining.pt')
        #scores.append(score)
    #x = [i + 1 for i in range(n_games)]
    #filename = 'lunar-laner.png'
    #plotLearning(x,scores,eps_history,filename)
