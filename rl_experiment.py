from rl_env import TinyEnv
from controllers import zero, pid, ppo

from stable_baselines3 import A2C, PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.env_checker import check_env

env = TinyEnv(model_path="./models/tinyphysics.onnx",data_path="./data/", controller=ppo.Controller)


# print(f'sample observation_space: {env.observation_space.sample()}')
# print(f'sample action_space: {env.action_space.sample()}')


# obs, info = env.reset()

# for _ in range(1000):
#     print(_)
#     sample_action = env.action_space.sample()
#     obs, reward, terminated, truncated, info = env.step(sample_action)
#     print(f"obs: {obs}\nreward: {reward}\nterminated: {terminated}\ntruncated: {truncated}\ninfo: {info}")
#     if terminated or truncated:
#         print('Env has terminated')
#         print('Resetting')
#         obs, info = env.reset()





# It will check your custom environment and output additional warnings if needed
# check_env(env)

# Define and Train the agent
model = PPO("MlpPolicy", env).learn(total_timesteps=1_000_000)


obs,info = env.reset()
for i in range(100):
    action, _state = model.predict(obs, deterministic=True)
    # print(f'action: {action}')
    obs, reward, terminated, truncated, info = env.step(action)
    # print(f'observation: {obs}')
    if terminated or truncated:
        obs, info = env.reset()
        # print()
        # print('Reset Env')
        # print()
    print(f'reward: {reward}')