# -*- encoding: utf-8 -*-
import chainer
import chainerrl
from model import Cource

def make_agent(env, obs_size, n_actions):
    """
    チュートリアル通りのagent作成
    ネットワークやアルゴリズムの決定
    """
    n_hidden_channels = 40
    n_hidden_layers = 5
    # 幅n_hidden_channels，隠れ層n_hidden_layersのネットワーク
    q_func = chainerrl.q_functions.FCStateQFunctionWithDiscreteAction(
        obs_size, n_actions, n_hidden_channels, n_hidden_layers
    )

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
        replay_start_size=500, update_frequency=1,
        target_update_frequency=100
    )
    return agent


def train_module(env, agent):
    """
    chainerrlのモジュールによるtraining
    """
    import logging
    import sys
    import gym
    gym.undo_logger_setup()  # Turn off gym's default logger settings
    logging.basicConfig(level=logging.INFO, stream=sys.stdout, format='')

    chainerrl.experiments.train_agent_with_evaluation(
        agent, env,
        steps=2000,           # 合計10000ステップagentを動かす
        eval_n_runs=5,         # 本番テストのたびに 5回評価を行う
        max_episode_len=200,   # 1ゲームのステップ数
        eval_frequency=1000,   # 1000ステップごとに本番テストを行う
        outdir='agent/result') # Save everything to 'agent/result' directory


def train_mine(env, agent):
    """
    自分でループを組むtraining
    1ゲームあたりmax_episode_lenの長さで
    n_episodes回訓練を行う
    """
    n_episodes = 50
    max_episode_len = 200
    log = []
    for i in range(1, n_episodes + 1):
        obs = env.reset()
        reward = 0
        done = False
        R = 0
        t = 0
        while not done and t < max_episode_len:
            action = agent.act_and_train(obs, reward)
            obs, reward, done, info = env.step(action)
            R += reward
            t += 1
            log.append(
                "acc=%d handle=%d car=%s rew=%f obs=%s" %
                (action&1, action&6, info, reward, str(obs))
            )
        if i % 10 == 0:
            print(
                'episode: ', i,
                'R:', R,
                'obs:', obs,
                'statistics:', agent.get_statistics()
            )
            env.render()
        agent.stop_episode_and_train(obs, reward, done)

    print('Finished')
    return log


def play(env, agent):
    """
    本番テスト
    """
    import canvas
    best_episode = []
    max_R = 0
    for i in range(10):
        obs = env.reset()
        done = False
        R = 0
        t = 0
        pos_tmp = []
        while not done and t < 200:
            # env.render()
            action = agent.act(obs)
            obs, r, done, _ = env.step(action)
            R += r
            t += 1
            pos_tmp.append(env.car.get_vec())
        print('test episode:', i, 'R:', R)
        agent.stop_episode()

        if R > max_R:
            max_R = R
            best_episode = pos_tmp

        canvas.draw(pos_tmp)
    # 最良エピソードを描画する (pygame使用)
    print('Finish demo')


if __name__ == '__main__':
    # 環境の作成
    env = Cource()

    obs_size = env.OBS_SIZE
    n_actions = env.ACTIONS
    agent = make_agent(env, obs_size, n_actions)

    save_path = 'agent/circle_2'
    # agent.load(save_path)

    # training
    train_module(env, agent)
    agent.save(save_path)

    # 訓練済みのagentを使ってテスト
    play(env, agent)
