from typing import Optional
from pathlib import Path
import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO
from tinyphysics import TinyPhysicsModel, TinyPhysicsSimulator
from tinyphysics import ACC_G, FPS, CONTEXT_LENGTH,CONTROL_START_IDX,COST_END_IDX, VOCAB_SIZE, LAT_ACCEL_COST_MULTIPLIER,LATACCEL_RANGE,STEER_RANGE, MAX_ACC_DELTA,DEL_T, FUTURE_PLAN_STEPS
from tinyphysics import FuturePlan

FUTURE_DIM = 10

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

        # add future plan to the observation space 
        self.observation_space = gym.spaces.Box(low=-100., 
                                                high=100., 
                                                shape=(4+FUTURE_DIM*4,), 
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

        # print(self.tiny_sim.futureplan)


        roll_lataccel, v_ego, a_ego = self.tiny_sim.state_history[-1]
        target_lat_accel = self.tiny_sim.target_lataccel_history[-1]
        
        obs_state = np.array([roll_lataccel, v_ego, a_ego, target_lat_accel])
        obs_state = np.reshape(obs_state, (-1,4))
        
        # these variables will not always be of correct shape 
        lataccel, roll, v, a = self.tiny_sim.futureplan
        obs_future = np.array([roll, v, a, lataccel])
        obs_future = np.transpose(obs_future)
        obs_future = obs_future[:FUTURE_DIM,:]

        if obs_future.shape[0] < FUTURE_DIM:
            row_pad_dim = FUTURE_DIM - obs_future.shape[0]
            zero_pad = np.zeros((row_pad_dim, 4))
            obs_future = np.vstack((obs_future, zero_pad))
        
        observation = np.concatenate((obs_state, obs_future))
        observation = observation.flatten()
        
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
        
        # observation = np.array([roll_lataccel, v_ego, a_ego, target_lat_accel])
        
        # NEW RESET OBS
        obs_state = np.array([roll_lataccel, v_ego, a_ego, target_lat_accel])
        obs_state = np.reshape(obs_state, (-1,4))
        
        obs_future = np.zeros((FUTURE_DIM,4))
        
        observation = np.concatenate((obs_state, obs_future))
        observation = observation.flatten()




        # NEW RESET OBS
        info = {'target_lat_accel':target_lat_accel}
        
        # if hasattr(FuturePlan, 'self.tiny_sim.futureplan'):
        #     print('yes, future plan present after reset')
        # else:
        #     print('no, future plan not present')



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
