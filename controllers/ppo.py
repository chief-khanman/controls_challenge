from . import BaseController
import numpy as np 

class Controller(BaseController):
    '''
    PPO controller 
    '''
    def __init__(self, controller_name = 'PPO'):
        super().__init__(controller_name)
        # define the database
        # define buffer
        
    def _policy_network(self,):
        pass

    def _value_network(self,):
        pass

    def update(target_lataccel, current_lataccel, state, future_plan):
        # use target_lataccel, current_lataccel, state -> input to policy and value network
        # action 
        # value
        # update buffers
        return None
    
