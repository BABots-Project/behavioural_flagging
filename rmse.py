import json

import numpy as np

from worm import state_dic

with open('state_estimations/worm_1/state_data.json') as f:
    true_approx_data = json.load(f)

true_states_ms = np.zeros(len(state_dic))
true_states_qs = np.zeros(len(state_dic))