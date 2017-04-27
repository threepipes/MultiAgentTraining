# -*- encoding: utf-8 -*-
import chainer
import chainerrl
from model import Cource
import chainer.links as L
import chainer.functions as F

import canvas


class QFunction(chainer.Chain):

    def __init__(self, n_actions):
        super(QFunction, self).__init__(
            conv1=L.Convolution2D(1, 5, 5, stride=1), # 60 -> 56 * 5
            conv2=L.Convolution2D(5, 2, 7), # -> 50 * 2
            # max pooling: -> 25 * 2
            l1=F.Linear(25 * 25 * 2, 400),
            l2=F.Linear(400, n_actions)
        )

    def __call__(self, x, test=False):
        h0 = F.relu(self.conv1(x))
        h1 = F.max_pooling_2d(F.relu(self.conv2(h0)), 2)
        h2 = F.dropout(F.relu(self.l1(h1)), train=(not test))
        return chainerrl.action_value.DiscreteActionValue(self.l2(h2))


def make_agent(env, obs_size, n_actions):
    """
    チュートリアル通りのagent作成
    ネットワークやアルゴリズムの決定
    """
    # n_hidden_channels = 800
    # n_hidden_layers = 4
    # # 幅n_hidden_channels，隠れ層n_hidden_layersのネットワーク
    # q_func = chainerrl.q_functions.FCStateQFunctionWithDiscreteAction(
    #     obs_size, n_actions, n_hidden_channels, n_hidden_layers
    # )
    q_func = QFunction(n_actions)

    q_func.to_gpu(0)

    # 最適化関数の設定
    optimizer = chainer.optimizers.Adam(1e-2)
    optimizer.setup(q_func)

    # 割引率の設定
    gamma = 0.95

    # 探索方針の設定
    explorer = chainerrl.explorers.ConstantEpsilonGreedy(
        epsilon=0.3, random_action_func=env.get_action_space()
    )

    replay_buffer = chainerrl.replay_buffer.ReplayBuffer(10 ** 6)

    agent = chainerrl.agents.DoubleDQN(
        q_func, optimizer, replay_buffer, gamma, explorer,
        replay_start_size=100, update_interval=1, gpu=0,
        target_update_interval=100
    )
    return agent


def train_mine(env, agent, agent_test):
    """
    自分でループを組むtraining
    1ゲームあたりmax_episode_lenの長さで
    n_episodes回訓練を行う
    """
    tmpfile = 'agent/tmp'
    n_episodes = 2000
    max_episode_len = 200
    n_agents = env.N_AGENTS
    log = []
    for i in range(1, n_episodes + 1):
        obs_list = env.reset()
        reward = [0] * n_agents
        done = False
        R = [0] * n_agents
        t = 0
        if i > 1:
            agent_test.load(tmpfile)
        while not done and t < max_episode_len:
            action_list = []
            for j, obs in enumerate(obs_list):
                if j == 0:
                    # canvas.draw_digit(obs[0], len(obs[0]))
                    action = agent.act_and_train(obs, reward[j])
                else:
                    action = agent_test.act(obs)
                    # action = 0
                action_list.append(action)
            environment = env.step(action_list)
            # obs, reward, done, info = env.step(action_list)
            for j, (obs, rew, done, info) in enumerate(environment):
                R[j] += rew
                reward[j] = rew
                obs_list[j] = obs
                log.append(
                    "car=%s rew=%f" %
                    (info, rew)
                )
            t += 1
            if t % 10 == 0:
                print(t)
        if i % 1 == 0:
            print(
                'episode: ', i,
                'R:', R,
                'statistics:', agent.get_statistics()
            )
            env.render()
        for obs, rew, done, info in environment[:1]:
            agent.stop_episode_and_train(obs, rew, done)
        agent_test.stop_episode()
        agent.save(tmpfile)

    print('Finished')
    return log



def play(env, agent):
    """
    自分でループを組むtraining
    1ゲームあたりmax_episode_lenの長さで
    n_episodes回訓練を行う
    """
    import canvas
    n_episodes = 10
    max_episode_len = 200
    n_agents = env.N_AGENTS
    log = []
    for i in range(1, n_episodes + 1):
        obs_list = env.reset()
        done = False
        R = [0] * n_agents
        t = 0
        pos_list = []
        while not done and t < max_episode_len:
            action_list = []
            for obs in obs_list:
                action = agent.act(obs)
                action_list.append(action)
            environment = env.step(action_list)
            t += 1
            for j, (obs, rew, done, info) in enumerate(environment):
                R[j] += rew
                obs_list[j] = obs
            pos_list.append(env.get_vecs())
            if t % 10 == 0:
                print(t)
        print(
            'episode: ', i,
            'R:', R,
            'statistics:', agent.get_statistics()
        )
        agent.stop_episode()
        canvas.draw(pos_list)

    print('Finished')
    return log


if __name__ == '__main__':
    # 環境の作成
    env = Cource()

    obs_size = env.OBS_SIZE
    n_actions = env.ACTIONS
    agent = make_agent(env, obs_size, n_actions)
    agent_test = make_agent(env, obs_size, n_actions)

    save_path = 'agent/circle_multi_conv'
    agent.load(save_path)

    # training
    train_mine(env, agent, agent_test)
    agent.save(save_path)

    # 訓練済みのagentを使ってテスト
    play(env, agent)
