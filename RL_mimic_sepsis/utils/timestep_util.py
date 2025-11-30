"""Timestep utils.
How to use:
sys.path.append('F:/time_step/OfflineRL_FactoredActions')
from RL_mimic_sepsis.utils.timestep_util import get_horizon

"""

def get_horizon(timestep):
    """
    """
    timestep_map = {1:80, 2:41, 4:21, 8:11}
    return timestep_map[timestep]

def get_state_dim(timestep: int, action_space: str) -> int:
    """Returns the state dimension based on timestep and action space.
    """
    return 128
    
timestep_list = [1, 2, 4, 8]
action_space_list = ['Quantile', 'Threshold', 'NormThreshold']
action_space_name_mapping = {'Quantile' : 'Quantile-5', 'Threshold' : 'Clinical-Threshold', 'NoreThreshold' : 'NormThreshold'}

