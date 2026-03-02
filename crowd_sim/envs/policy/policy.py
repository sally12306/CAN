import abc
import numpy as np


class Policy(object):
    def __init__(self):
        """
        Base class for all policies, has an abstract method predict().
        """
        self.trainable = False
        self.phase = None
        self.model = None
        self.device = None
        self.last_state = None
        self.time_step = None
        # if agent is assumed to know the dynamics of real world
        self.env = None

    @abc.abstractmethod
    def configure(self, config):
        return

    def set_phase(self, phase):
        self.phase = phase

    def set_device(self, device):
        self.device = device

    def set_env(self, env):
        self.env = env

    def get_model(self):
        return self.model

    @abc.abstractmethod
    def predict(self, state):
        """
        Policy takes state as input and output an action

        """
        return

    @staticmethod
    def reach_destination(state):
        """
        使用 `np.linalg.norm` 函数计算两个向量的范数
        函数将计算得到的距离与机器人自身状态的半径进行比较。
        如果距离小于机器人的半径，说明机器人已经到达目的地，函数返回 `True
        """
        self_state = state.self_state
        if np.linalg.norm((self_state.py - self_state.gy, self_state.px - self_state.gx)) < self_state.radius:
            return True
        else:
            return False
