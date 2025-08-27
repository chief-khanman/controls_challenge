from rl_env import TinyEnv
from controllers import zero, pid, ppo


env = TinyEnv(model_path="./models/tinyphysics.onnx",data_path="./data/", controller=ppo.Controller)
print(f'sample observation_space: {env.observation_space.sample()}')
print(f'sample action_space: {env.action_space.sample()}')


obs, info = env.reset()
for _ in range(105):
    sample_action = env.action_space.sample().item() #! problem - the output of sample is an array - needs to be float 
    obs, reward, terminated, truncated, info = env.step(sample_action)
    print(f"obs: {obs}\nreward: {reward}\nterminated: {terminated}\ntruncated: {truncated}\ninfo: {info}")