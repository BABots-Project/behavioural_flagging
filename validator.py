import json
import os.path

import matplotlib.pyplot as plt
import numpy as np
import scipy.special as sp

from worm import state_dic


def compute_msd(x, y):
    """
    Compute the mean squared displacement (MSD) of a single trajectory.
    """
    N = len(x)
    msd = []
    for tau in range(1, N):  # Lag time
        displacements = [(x[t + tau] - x[t]) ** 2 + (y[t + tau] - y[t]) ** 2 for t in range(N - tau)]
        msd.append(np.mean(displacements))
    return msd

def correct_for_pbc(xi_list, yi_list, Lx, Ly):
    """
    Correct trajectories for periodic boundary conditions.

    Parameters:
        xi_list (list): x-positions of an agent over time.
        yi_list (list): y-positions of an agent over time.
        Lx (float): Length of the domain in the x-direction.
        Ly (float): Length of the domain in the y-direction.

    Returns:
        unwrapped_x (list): Corrected x-positions.
        unwrapped_y (list): Corrected y-positions.
    """
    unwrapped_x = [xi_list[0]]  # Start with the initial position
    unwrapped_y = [yi_list[0]]

    for i in range(1, len(xi_list)):
        dx = xi_list[i] - xi_list[i - 1]
        dy = yi_list[i] - yi_list[i - 1]

        # Correct for PBC in x-direction
        if dx > Lx / 2:
            dx -= Lx
        elif dx < -Lx / 2:
            dx += Lx

        # Correct for PBC in y-direction
        if dy > Ly / 2:
            dy -= Ly
        elif dy < -Ly / 2:
            dy += Ly

        # Append the unwrapped positions
        unwrapped_x.append(unwrapped_x[-1] + dx)
        unwrapped_y.append(unwrapped_y[-1] + dy)

    return unwrapped_x, unwrapped_y

def compute_msd_with_pbc(xi_list, yi_list, Lx, Ly):
    """
    Compute MSD for a single trajectory, accounting for periodic boundary conditions.

    Parameters:
        xi_list (list): x-positions of an agent over time.
        yi_list (list): y-positions of an agent over time.
        Lx (float): Length of the domain in the x-direction.
        Ly (float): Length of the domain in the y-direction.

    Returns:
        msd (list): MSD as a function of lag time.
    """
    # Unwrap trajectories
    unwrapped_x, unwrapped_y = correct_for_pbc(xi_list, yi_list, Lx, Ly)

    # Calculate MSD
    n = len(unwrapped_x)
    msd = []
    for lag in range(1, n):
        squared_displacements = [
            (unwrapped_x[i + lag] - unwrapped_x[i]) ** 2 +
            (unwrapped_y[i + lag] - unwrapped_y[i]) ** 2
            for i in range(n - lag)
        ]
        msd.append(np.mean(squared_displacements))
    return msd

worm_positions = np.load("states/worm_1_6_positions.npy")
worm_states = np.load("states/worm_1_6_states.npy")[::4]
worm_xs = worm_positions[0][::4]
worm_ys = worm_positions[1][::4]

worm_msd = compute_msd(worm_xs, worm_ys)

with open("/home/nema/cuda_agent_based_sim/agents_all_data.json") as f:
    agent_all_data = json.load(f)

position_matrix = []
msd_matrix = []
n_agents = agent_all_data["parameters"]["N"]
print(n_agents)
n_steps = agent_all_data["parameters"]["N_STEPS"]


if os.path.isfile("tmp_agent_msd_matrix.npy"):
    msd_matrix = np.load("tmp_agent_msd_matrix.npy")
    for i in range(n_agents):
        agent_msd = msd_matrix[i]
        plt.plot(np.arange(len(agent_msd)), np.log10(agent_msd), c='orange', alpha=0.6, label="agent")
else:
    for i in range(n_agents):
        print(i)
        positions = agent_all_data["positions"][i:i+n_steps][0]
        lx = agent_all_data["parameters"]["WIDTH"]*1e3
        ly = agent_all_data["parameters"]["HEIGHT"]*1e3

        xs = [pos[0]*1e3 for pos in positions]
        ys = [pos[1]*1e3 for pos in positions]
        agent_msd = compute_msd_with_pbc(xs, ys, lx, ly)
        msd_matrix.append(agent_msd)
        plt.plot(np.arange(len(agent_msd)), np.log10(agent_msd), c='orange', alpha=0.6, label="agent")

    np.save("tmp_agent_msd_matrix", msd_matrix)
plt.plot(range(len(worm_msd)), np.log10(worm_msd), label="worm", c="blue")
plt.title("MSD over time")

duration=1
estimated_msds = []
with open('state_estimations/worm_1/state_data.json') as f:
    state_properties_dic = json.load(f)
print(state_properties_dic)

def autocorrelation(n, state_index):
    state = list(state_dic.keys())[state_index]
    v_alpha_t = state_properties_dic[state]["speed"]["alpha"]
    v_beta_t = state_properties_dic[state]["speed"]["beta"]
    v_state_scale = state_properties_dic[state]["speed"]["scale"]
    v_sq = (v_alpha_t / (v_beta_t + v_alpha_t) * v_state_scale) ** 2
    kappa_t = state_properties_dic[state]["angle"]["kappa"]
    mu_t = state_properties_dic[state]["angle"]["mu"]
    rho = sp.i1(kappa_t)/sp.i0(kappa_t)
    return v_sq * np.cos(n*mu_t) * rho ** n

start_index = 0
for t, cur_state in enumerate(worm_states[:-1]):
    next_state = worm_states[t+1]
    if cur_state == next_state:
        duration+=1
    else:
        summation = 0
        for n in range(1, duration):
            summation+= (duration-n) * autocorrelation(n, int(cur_state))
        delta_msd = 2 * summation
        estimated_msd = duration * autocorrelation(0, int(cur_state)) + delta_msd #int(worm_states[start_index])
        start_index = t
        estimated_msds.append(estimated_msd)
        duration = 1

#plt.plot(range(len(estimated_msds)), np.log10(estimated_msds), label="estimation", c="black")
n_steps = len(worm_states)
cumulative_disp = np.array([0.0, 0.0])  # Initialize the displacement vector.
current_angle = 0.0  # Start with an initial angle of 0.
msds = []  # List to store MSD at each timestep.

for t in range(n_steps):
    # Get the current state's index (convert to int if needed)
    state_index = int(worm_states[t])
    # Retrieve the state name using your state_dic mapping.
    state = list(state_dic.keys())[state_index]

    # Extract the speed parameters and compute the expected speed.
    v_alpha = state_properties_dic[state]["speed"]["alpha"]
    v_beta = state_properties_dic[state]["speed"]["beta"]
    v_scale = state_properties_dic[state]["speed"]["scale"]
    expected_v = (v_alpha / (v_alpha + v_beta)) * v_scale

    # Extract the expected turning angle increment (mu) for this state.
    expected_mu = state_properties_dic[state]["angle"]["mu"]
    kappa_t = state_properties_dic[state]["angle"]["kappa"]
    rho = sp.i1(kappa_t)/sp.i0(kappa_t)

    # Update the cumulative angle by adding the expected turning angle.
    current_angle += expected_mu

    # Compute the displacement increment from the expected speed and current angle.
    dx = expected_v * np.cos(current_angle) * rho
    dy = expected_v * np.sin(current_angle) * rho
    increment = np.array([dx, dy])

    # Update the cumulative displacement.
    cumulative_disp += increment

    # Compute the MSD as the square of the displacement magnitude.
    msd = cumulative_disp[0] ** 2 + cumulative_disp[1] ** 2
    msds.append(msd)

# Plot the log10(MSD) over time.
plt.plot(range(n_steps), np.log10(msds), label="estimated", c="black")
plt.xlabel("Timestep")
plt.ylabel("log10(MSD)")


plt.legend()
plt.show()
