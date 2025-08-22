import gymnasium as gym
from stable_baselines3 import PPO
from tinyphysics import TinyPhysicsModel, TinyPhysicsSimulator




class TinyEnv(gym.Env):
    def __init__(self):
        super().__init__()
        self.tiny_sim = TinyPhysicsSimulator(TinyPhysicsModel, data_path=None, controller=)

    def step(self,action):
        return observation, reward, terminated, truncated, info

    def reset(self,):
        return observation, info

    def render(self,):
        pass

    def close(self,):
        pass