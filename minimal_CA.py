#!/usr/bin/env python

import argparse
import json
import logging
import sys
import time

import brica1
import brica1.brica_gym
import numpy as np
from gym import wrappers
from oculoenv import Environment as OculoEnvironment
from oculoenv import PointToTargetContent

import myenv  # do not delete this line. This module is indirectory refered by gym.make.
from modules.CognitiveArchitecture import CognitiveArchitecture

if __name__ == '__main__':

    # 引数の処理
    parser = argparse.ArgumentParser(description='BriCA Minimal Cognitive Architecture with Gym')
    parser.add_argument('mode', help='1:random act, 2: reinforcement learning', choices=['1', '2'])
    parser.add_argument('--dump', help='dump file path')
    parser.add_argument('--episode_count', type=int, default=1, metavar='N',
                      help='Number of training episodes (default: 1)')
    parser.add_argument('--max_steps', type=int, default=20, metavar='N',
                      help='Max steps in an episode (default: 20)')
    parser.add_argument('--config', type=str, default='minimal_CA.json', metavar='N',
                      help='Model configuration (default: minimal_CA.json')
    parser.add_argument('--model', type=str, metavar='N',
                      help='Saved model for retina path')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                      help='disables CUDA training')
    args = parser.parse_args()

    # Configファイルの読み込み
    with open(args.config) as config_file:
       config = json.load(config_file)

    # ロガー設定
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # 引数にDumpが指定されている場合は、そこに出力するよう設定
    if args.dump is not None:
        try:
            observation_dump = open(args.dump, mode='w')
        except:
            print('Error: No dump path specified', file=sys.stderr)
            sys.exit(1)
    else:
        observation_dump = None
    
    # OculoEnvのEnvironmentの設定（タスク定義はOculoEnv側に定義し、それをここで指定する。その他設定もConfigから読み込み、ここで設定）
    content = PointToTargetContent()
    env = OculoEnvironment(content)
    outdir = config['gym_monitor_outdir']
    env = wrappers.Monitor(env, outdir, force=True)
    train={}
    train["episode_count"]=args.episode_count
    train["max_steps"]=args.max_steps

    # OculoEnvのAgentの設定
    if args.mode == "1":   # random act
        model = CognitiveArchitecture(False, train, False, config=config)
    elif args.mode == "2": # act by reinforcement learning
        train['rl_agent']=config['rl_agent']
        modelp = args.model is not None
        model = CognitiveArchitecture(True, train, modelp, config=config)
    agent = brica1.brica_gym.GymAgent(model, env)
    scheduler = brica1.VirtualTimeSyncScheduler(agent)
    
    # 学習のイテレーションを回す処理
    last_token = 0
    for i in range( train["episode_count"]):
        reward_sum = 0.
        for j in range(train["max_steps"]):
            scheduler.step()
            model.retina.inputs['token_in'] = model.get_out_port('token_out').buffer
            time.sleep(config["sleep"])
            current_token = agent.get_out_port('token_out').buffer[0]
            if last_token + 1 == current_token:
                reward_sum += agent.get_in_port("reward").buffer[0]
                last_token = current_token
                env.render()
                if observation_dump is not None:
                     observation_dump.write(str(agent.get_in_port("observation").buffer.tolist()) + '\n')
            if agent.env.done:
                agent.env.flush = True
                scheduler.step()
                while agent.get_in_port('token_in').buffer[0]!= agent.get_out_port('token_out').buffer[0]:
                    scheduler.step()
                agent.env.reset()
                model.fef.reset()
                model.retina.results['token_out'] = np.array([0])
                model.retina.out_ports['token_out'].buffer = np.array([0])
                last_token = current_token = 0
                break
        print(i, "Avr. reward: ", reward_sum)

    print("Close")
    if observation_dump is not None:
        observation_dump.close()
    env.close()
