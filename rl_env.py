from typing import Optional
from pathlib import Path
import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO
from tinyphysics import TinyPhysicsModel, TinyPhysicsSimulator
from tinyphysics import ACC_G, FPS, CONTEXT_LENGTH,CONTROL_START_IDX,COST_END_IDX, VOCAB_SIZE, LAT_ACCEL_COST_MULTIPLIER,LATACCEL_RANGE,STEER_RANGE, MAX_ACC_DELTA,DEL_T, FUTURE_PLAN_STEPS
from tinyphysics import FuturePlan

FUTURE_DIM = 10



class EpisodeSampler:
    def __init__(self, data_dir: str, data_split:float = 0.7, test=False, seed=123):
        
        self.test = test
        self.seed = seed
        rng = np.random.default_rng(seed=self.seed)
        self.files = [str(p) for p in sorted(Path(data_dir).iterdir())]
        rng.shuffle(self.files)
        assert len(self.files) > 0, "No CSV files found."
        
        
        # split files into two lists
        splitindex = int(len(self.files)//(data_split*10))
        # training list
        self.train_files = self.files[splitindex:]
        self.train_idx = 0
        # next should use training_list  
        
        # testing list
        self.test_files = self.files[:splitindex]
        self.test_idx = 0
        # lets introduce another bool argument - 'test = True/False'
        # if test=True, then next will use test_files list 
        self._shuffle()


    def _shuffle(self):
        if self.test == True:
            rng = np.random.default_rng(self.seed)
            rng.shuffle(self.test_files)
            self.test_idx = 0
        else:
            rng = np.random.default_rng(self.seed)
            rng.shuffle(self.train_files)
            self.train_idx = 0

    def next(self):
        # TEST
        if self.test == True:
            if self.test_idx >= len(self.test_files):
                self._shuffle()
            f = self.test_files[self.test_idx]
            self.test_idx += 1
            return f
        # TRAIN
        else:
            if self.train_idx >= len(self.train_files):
                self._shuffle()
            f = self.train_files[self.train_idx]
            self.train_idx += 1
            return f





class TinyEnv(gym.Env):
    def __init__(self,
                 model_path = None, 
                 data_path=None, 
                 controller=None, 
                 debug = False):
        
        super().__init__()
        
        #updating/fixing reset 
        self.episode_sampler = EpisodeSampler(data_dir=data_path)
        
        
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
        
        # self.data_path = Path(data_path) # path to dir
        # self.data_files = sorted(self.data_path.iterdir())
        
    
    
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
        
        pred = self.tiny_sim.current_lataccel

        #cost = self._get_reward(target=target_lat_accel, pred=pred)
        
        reward = self._get_reward(target=target_lat_accel, pred=pred, action=action)
        
        truncated = False
        
        #! when a .csv file reaches its end we will send terminated signal as TRUE
        terminated = True if self.tiny_sim.step_idx == len(self.tiny_sim.data) else False
        
        info = {'target_lat_accel': target_lat_accel}
        
        return observation, reward, terminated, truncated, info

    
    
    def reset(self, options:Optional[dict]=None, **kwargs):

        # action history
        #set data_file for step 
        self.data_file = self.episode_sampler.next()
        print(self.data_file)
        #create a tiny_sim
        self.tiny_sim = TinyPhysicsSimulator(model=self.tiny_model,
                                             data_path=self.data_file,
                                             controller=self.controller,
                                             debug=self.debug)
        
        self.tiny_sim.reset()
        
        self.prev_action = self.tiny_sim.action_history[-1] if self.tiny_sim.action_history else 0.0
        
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


    def _get_reward(self,target, pred, action):
        '''Compute and return the reward for each step of env'''
        preview_len: int = 50
        jerk_w =           0.01
        act_w    =         0.003
        
        error = (pred - target)
        act = self.tiny_sim.action_history[-1]
        dact = act - self.prev_action
        #       
        reward = -error*error - jerk_w*(dact*dact) - act_w*(act*act)
        self.prev_action = action
        return reward
