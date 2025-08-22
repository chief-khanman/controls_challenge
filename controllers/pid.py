from . import BaseController
import numpy as np

class Controller(BaseController):
  """
  A simple PID controller
  """
  def __init__(self,):
    self.p = 0.3
    self.i = 0.05
    self.d = -0.1
    self.error_integral = 0
    self.prev_error = 0

  def update(self, target_lataccel, current_lataccel, state, future_plan):
      '''For PID - 
      Args: 
        target_lataccel - this information comes from a planner(here its the data, in real world its the planner)
        current_lataccel
      Return:
        control_action'''
      error = (target_lataccel - current_lataccel)
      self.error_integral += error
      error_diff = error - self.prev_error
      self.prev_error = error
      control_action = (self.p * error 
                      + self.i * self.error_integral 
                      + self.d * error_diff)
      return control_action
