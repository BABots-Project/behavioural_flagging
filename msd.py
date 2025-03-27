import json
import os

import numpy as np
from matplotlib import pyplot as plt

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


with open("/home/nema/cuda_agent_based_sim/agents_all_data.json") as f:
    agent_all_data = json.load(f)

position_matrix = []
msd_matrix = []
n_agents = agent_all_data["parameters"]["N"]
print(n_agents)
n_steps = agent_all_data["parameters"]["N_STEPS"]

msd_target_file = f'tmp_agent_msd_matrix_n_{n_agents}.npy'

if os.path.isfile(msd_target_file):
    msd_matrix = np.load(msd_target_file)
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

    np.save(msd_target_file.split(".")[0], msd_matrix)

plt.show()
