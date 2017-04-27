import sys
import json
import matplotlib.pyplot as plt
import numpy as np


def load_log_json(filepath):
    with open(filepath) as f:
        data = json.load(f)
    reward_list = []
    loss_list = []
    for row in data:
        reward_list.append(row['reward'])
        loss_list.append(row['loss'])
    return reward_list, loss_list


def plot_rewards(reward_list):
    turn = len(reward_list)
    y = [[] for i in range(len(reward_list[0]))]
    for rewards in reward_list:
        for i, rew in enumerate(sorted(rewards)):
            y[i].append(rew)
    x = np.arange(turn)
    for y_data in y:
        plt.plot(x, np.array(y_data))
    plt.show()


def plot_loss(loss_list):
    plt.plot(np.arange(len(loss_list), np.array(loss_list)))
    plt.show()


if __name__ == '__main__':
    if len(sys.argv) > 1:
        filename = 'plot/' + sys.argv[1]
    else:
        filename = 'plot/tmp'

    rews, losses = load_log_json(filename)
    plot_rewards(rews)
    plot_loss(losses)
    