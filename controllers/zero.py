from . import BaseController


class Controller(BaseController):
  """
  A controller that always outputs zero
  """
  def __init__(self, controller_name = 'Zero'):
    super().__init__(controller_name)
  def update(self, target_lataccel, current_lataccel, state, future_plan):
    return 0.0
