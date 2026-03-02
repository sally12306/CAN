import logging
import argparse
import configparser
import os
import torch
import numpy as np
import gym
# from crowd_nav.policy.policy_factory import policy_factory
from crowd_sim.envs.utils.robot import Robot
from crowd_sim.envs.policy.orca import ORCA
from crowd_sim.envs.utils.info import *
from crowd_sim.envs.utils.state import JointState
from crowd_sim.envs.utils.action import ActionRot, ActionXY
import h5py
from math import sqrt, pow
import queue
from collections import deque


def main():
    parser = argparse.ArgumentParser('Parse configuration file')
    parser.add_argument('--env_config', type=str,
                        default='D:\Program Files (x86)\IDE\JetBrains\PycharmProjects\CrowdNav-master_danren\crowd_nav\configs\env.config')
    parser.add_argument('--policy_config', type=str,
                        default='D:\Program Files (x86)\IDE\JetBrains\PycharmProjects\CrowdNav-master_danren\crowd_nav\configs\policy.config')
    parser.add_argument('--policy', type=str, default='orca')
    parser.add_argument('--phase', type=str, default='train')
    parser.add_argument('--mode', type=str, default='d')
    args = parser.parse_args()

    env_config_file = args.env_config
    policy_config_file = args.policy_config

    # configure logging and device
    logging.basicConfig(level=logging.INFO, format='%(asctime)s, %(levelname)s: %(message)s',
                        datefmt="%Y-%m-%d %H:%M:%S")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logging.info('Using device: %s', device)

    # configure policy

    policy = ORCA()
    policy_config = configparser.RawConfigParser()
    policy_config.read(policy_config_file)
    policy.configure(policy_config)
    if args.policy == 'orca':
        policy.safety_space = 0.05

    # configure environment
    env_config = configparser.RawConfigParser()
    env_config.read(env_config_file)
    env = gym.make('CrowdSim-v0')
    env.configure(env_config)

    robot = Robot(env_config, 'robot')
    robot.set_policy(policy)
    env.set_robot(robot)

    policy.set_phase(args.phase)
    policy.set_device(device)

    policy.set_env(env)
    robot.policy.set_phase(args.phase)
    robot.print_info()

    run = True
    observations = []
    actions = []
    rewards = []
    terminals = []
    timeouts = []
    too_close = 0
    min_dist = []
    next_observations = []
    last_observations = []
    last_last_observations = []
    next_next_observations = []

    observations_7 = []
    observations_6 = []
    observations_5 = []
    observations_4 = []
    observations_3 = []

    cumulative_rewards = []
    success_times = []
    collision_times = []
    timeout_times = []

    # while True:
    sample = False
    success = 0
    total = 0
    collision = 0
    timeout = 0
    step = 0
    while run:
    #for _ in range(100):
        reward01 = []
        j = 0
        last_state = []
        next_next_state = []
        # print(total)
        # get human obs
        ob = env.reset(phase=args.phase)
        deque_list = deque(maxlen=8)
        deque_list.clear()
        # joint_state = JointState(robot.get_full_state(), ob)
        # state = to_np(robot.policy.transform(joint_state, mode=args.mode).view(1, -1).squeeze(0))
        # for i in range(7):
        #     deque_list.append(state)

        done = False
        while not done:
            action_xy = robot.act(ob)
            # u1 = np.random.random()
            # if u1 <= 1:
            vx = action_xy.vx + np.random.normal(0, 0.1)
            vx = clamp_xy(vx, -1, 1)

            vy = action_xy.vy + np.random.normal(0, 0.1)
            vy = clamp_xy(vy, -1, 1)
            action_xy = ActionXY(vx, vy)
            # print((vx ** 2+ vy**2)**0.5)

            joint_state = JointState(robot.get_full_state(), ob)
            state = to_np(robot.policy.transform(joint_state).view(1, -1).squeeze(0))
            # print(state.shape)
            # deque_list.append(state)

            # print(state[11::13])
            # print(state[11::13])

            # last_state.append(state)
            ob, reward, done, info = env.step(action_xy)
            # print(reward)

            new_joint_state = JointState(robot.get_full_state(), ob)
            next_state = to_np(robot.policy.transform(new_joint_state).view(1, -1).squeeze(0))
            # next_next_state.append(next_state)

            action = np.array([action_xy.vx, action_xy.vy])

            # if j == 0:
            #     last_observations.append(last_state[0])
            #     last_last_observations.append(last_state[0])
            # elif j == 1:
            #     last_observations.append(last_state[0])
            #     last_last_observations.append(last_state[0])
            #
            # else:
            #     last_observations.append(last_state[j - 1])
            #     last_last_observations.append(last_state[j - 2])
            # print('last',last_state[j - 1])
            # print('state', state)
            # print('next', next_state)
            # print(len(last_observations))
            # j += 1

            # env.render(mode='video')
            observations.append(state)
            next_observations.append(next_state)
            actions.append(action)
            rewards.append(reward)
            terminals.append(int(done))
            # observations_7.append(deque_list[0])
            # observations_6.append(deque_list[1])
            # observations_5.append(deque_list[2])
            # observations_4.append(deque_list[3])
            # observations_3.append(deque_list[4])

            # print(deque_list[0][:5])
            # print(deque_list[1][:5])
            # print(deque_list[2][:5])
            j += 1
            # print(j)

            reward01.append(reward)
            step += 1

            # if isinstance(info, Timeout):
            #     print('timeouts')
            #     timeouts.append(1)
            # else:
            #     timeouts.append(0)

            if isinstance(info, Danger):
                too_close += 1
                min_dist.append(info.min_dist)

            if len(observations) % 1000 == 0:
                print(len(observations))

            if len(observations) == 500000:
                run = False
                break

        #env.render(mode='traj')

        if isinstance(info, ReachGoal):
            success += 1
            total += 1
            success_times.append(env.global_time)
        elif isinstance(info, Collision):
            collision += 1
            total += 1
            collision_times.append(env.global_time)
        elif isinstance(info, Timeout):
            timeout += 1
            total += 1
            timeout_times.append(env.time_limit)

        # cumulative_rewards.append(sum(reward01))
        cumulative_rewards.append(sum([pow(0.9, t * 0.25)
                                       * reward for t, reward in enumerate(reward01)]))

    num_step = sum(success_times + collision_times + timeout_times) / 0.25
    print(num_step)
    # print(actions)

    # TODO
    print(success)
    print(total)
    print(timeout)
    print(success / total)
    avg_nav_time = sum(success_times) / len(success_times) if success_times else env.time_limit
    print(avg_nav_time)
    print(avg(cumulative_rewards))
    print(collision / total)
    print(too_close / num_step)
    print(avg(min_dist))

    print('step', step)

    file = 'Crowd_nav_5'
    file_name = '{}/{}'.format(file, file) + '.hdf5'

    if not os.path.exists(file):

        print('start make data')

        os.mkdir(file)

        f = h5py.File(file_name, "w")
        # 和python打开文件的方式一样，可以有'w',有'a'

        # 创建一个dataset
        # observations_7 = f.create_dataset("observations_7", data=observations_7)
        # observations_6 = f.create_dataset("observations_6", data=observations_6)
        # observations_5 = f.create_dataset("observations_5", data=observations_5)
        # observations_4 = f.create_dataset("observations_4", data=observations_4)
        # observations_3 = f.create_dataset("observations_3", data=observations_3)
        # last_last_observations = f.create_dataset("last_last_observations", data=last_last_observations)
        # last_observations = f.create_dataset("last_observations", data=last_observations)
        observations = f.create_dataset("observations", data=observations)
        next_observations = f.create_dataset("next_observations", data=next_observations)
        actions = f.create_dataset("actions", data=actions)
        rewards = f.create_dataset("rewards", data=np.array(rewards))
        terminals = f.create_dataset("terminals", data=np.array(terminals))
        timeouts = f.create_dataset("timeouts", data=np.array(timeouts))

        f.close()

        if os.path.exists(file_name):
            print('make success')
        else:
            print('make fail')
    else:
        print('finish')


def avg(input_list):
    if input_list:
        return sum(input_list) / len(input_list)
    else:
        return


def to_np(t):
    if t is None:
        return None
    elif t.nelement() == 0:
        return np.array([])
    else:
        return t.cpu().detach().numpy()


def clamp_xy(num, min, max):
    if num < min:
        num = min
    elif num > max:
        num = max
    else:
        pass
    return num


if __name__ == '__main__':
    main()
