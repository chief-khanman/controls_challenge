from typing import Optional
from pathlib import Path
import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO
from tinyphysics import TinyPhysicsModel, TinyPhysicsSimulator
from tinyphysics import ACC_G, FPS, CONTEXT_LENGTH,CONTROL_START_IDX,COST_END_IDX, VOCAB_SIZE, LAT_ACCEL_COST_MULTIPLIER,LATACCEL_RANGE,STEER_RANGE, MAX_ACC_DELTA,DEL_T, FUTURE_PLAN_STEPS



class TinyEnv(gym.Env):
    def __init__(self,
                 model_path = None, 
                 data_path=None, 
                 controller=None, 
                 debug = False):
        
        super().__init__()
        
        self.debug = debug
        
        self.tiny_model = TinyPhysicsModel(model_path=model_path,
                                           debug=self.debug)
        
        self.controller = controller

        self.observation_space = gym.spaces.Box(low=np.array([-100,-100,-100,-100]), 
                                                high=np.array([100,100,100,100]), 
                                                shape=(4,), 
                                                dtype=np.float64)
        self.action_space = gym.spaces.Box(low=-1, 
                                           high=1, 
                                           shape=(1,), 
                                           dtype=np.float32)
        
        self.data_path = Path(data_path) # path to dir
        self.data_files = sorted(self.data_path.iterdir())
        
    
    
    def step(self,action:np.ndarray):
        action:float = action.item()
        self.tiny_sim.step(action)

        roll_lataccel, v_ego, a_ego = self.tiny_sim.state_history[-1]
        target_lat_accel = self.tiny_sim.target_lataccel_history[-1]
        observation = np.array([roll_lataccel, v_ego, a_ego, target_lat_accel])
        
        pred = self.tiny_sim.current_lataccel_history[-1]

        cost = self._get_reward(target=target_lat_accel, pred=pred)
        
        reward = -1.0 * cost
        
        truncated = False
        
        #! when a .csv file reaches its end we will send terminated signal as TRUE
        terminated = True if self.tiny_sim.step_idx == len(self.tiny_sim.data) else False
        
        info = {'target_lat_accel': target_lat_accel}
        
        return observation, reward, terminated, truncated, info

    
    
    def reset(self, options:Optional[dict]=None, **kwargs):
        #set data_file for step 
        self.data_file = str(self.data_files.pop(0))
        print(self.data_file)
        #create a tiny_sim
        self.tiny_sim = TinyPhysicsSimulator(model=self.tiny_model,
                                             data_path=self.data_file,
                                             controller=self.controller,
                                             debug=self.debug)
        
        self.tiny_sim.reset()
        
        roll_lataccel, v_ego, a_ego = self.tiny_sim.state_history[-1]
        target_lat_accel = self.tiny_sim.target_lataccel_history[-1]
        
        observation = np.array([roll_lataccel, v_ego, a_ego, target_lat_accel])
        
        info = {'target_lat_accel':target_lat_accel}
        
        return observation, info

    def render(self,):
        pass

    def close(self,):
        pass


    def _get_reward(self,target, pred):
        '''Compute and return the reward for each step of env'''
        lat_accel_cost = np.mean((target - pred)**2) * 100
        jerk_cost = np.mean((pred / DEL_T)**2) * 100
        total_cost = (lat_accel_cost * LAT_ACCEL_COST_MULTIPLIER) + jerk_cost
        return total_cost
