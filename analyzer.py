import json, sys

import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as opt
from scipy.stats import median_abs_deviation
from main_analyzer import analyze
from runner import frame_folder
from video_maker import animate_real_worm_with_state_changes
#frame_folder = "worm_1_6_"
from worm import state_dic

subsampling_rate=1
worm_velocities = np.load(f"states/{frame_folder}/velocities.npy")[::subsampling_rate]
#find outliers (above 300), they will be eradicated
outlier_indices = np.where(worm_velocities>300*subsampling_rate)[0]
outlier_mask = np.ones(len(worm_velocities), dtype=bool)
outlier_mask[outlier_indices] = False
worm_states = np.load(f"states/{frame_folder}/states.npy")[::subsampling_rate][outlier_mask]
worm_positions = np.load(f"states/{frame_folder}/positions.npy")
xs = np.array([x for x in worm_positions[0]][::subsampling_rate])[outlier_mask]
ys = np.array([x for x in worm_positions[1]][::subsampling_rate])[outlier_mask]
worm_positions = [xs, ys]
worm_velocities = worm_velocities[outlier_mask]

#animate_real_worm_with_state_changes(xs, ys, worm_states, worm_velocities)

with open("/home/nema/cuda_agent_based_sim/agents_all_data.json", "r") as f:
    agent_data = json.load(f)
#sys.exit()
agent_positions = agent_data["positions"]
agent_xs = [a[0] for a in agent_positions]
agent_ys = [a[1] for a in agent_positions]
agent_positions = np.array([agent_xs, agent_ys])
agent_states = np.array(agent_data["sub_states"])[0]
#agent_states = np.roll(agent_states, -1)
agent_velocities = np.array(agent_data["velocities"])[0]
agent_angles = np.array(agent_data["angles"])[0]

worm_angle_changes, worm_durations, worm_state_transitions, worm_events_per_time_window, worm_transition_fitting_functions, worm_time_state_transitions, worm_time_transition_fitting_functions = analyze(worm_states, worm_positions, True, velocities=worm_velocities, log_state_data=True, fps=1)
agent_angle_changes, agent_durations, agent_state_transitions, agent_events_per_time_window, agent_transition_fitting_functions, agent_time_state_transitions, agent_time_transition_fitting_functions = analyze(agent_states, agent_positions, False, angles=agent_angles, velocities=agent_velocities, target="Agent", fps=1)


fig, axs = plt.subplots(2, 2, figsize=(8, 8))
purified_worm_velocities =  [w for w in worm_velocities if w<=300]
axs[0,0].hist(purified_worm_velocities, bins=20)
axs[0,0].set_title("Real worm velocities")
axs[0,1].hist(agent_velocities, bins=20)
axs[0,1].set_title("Agent velocities")
axs[1,0].hist(worm_angle_changes, bins=20)
axs[1,0].set_title("Real worm angle changes")
axs[1,1].hist(agent_angle_changes, bins=20)
axs[1,1].set_title("Agent angle changes")
plt.savefig(f"states/{frame_folder}/overall_v_da.png")
plt.show()

duration_scale_fit_a = np.zeros(len(state_dic))
duration_scale_fit_b = np.zeros(len(state_dic))
duration_scale_fit_c = np.zeros(len(state_dic))

for i, key in enumerate(state_dic.keys()):
    fig, axs = plt.subplots(2, 4, figsize=(16, 8))
    agent_state_indices = np.where(agent_states == state_dic[key])[0]
    agent_state_angle_changes = agent_angle_changes[agent_state_indices]
    agent_state_velocities = agent_velocities[agent_state_indices]

    worm_state_indices = np.where(worm_states == state_dic[key])[0]
    worm_state_indices = worm_state_indices[np.where(worm_state_indices<len(worm_velocities))]
    worm_state_angle_changes = worm_angle_changes[worm_state_indices]
    worm_state_velocities = np.array(worm_velocities)[worm_state_indices]

    axs[0, 0].hist(worm_state_angle_changes, bins=20)
    axs[0, 0].set_title(f'Angle changes for state {key} | Real worm')
    axs[1, 0].hist(agent_state_angle_changes, bins=20)
    axs[1, 0].set_title(f'Angle changes for state {key} | Agent')

    axs[0, 1].hist(worm_state_velocities, bins=20)
    axs[0, 1].set_title(f'Velocity for state {key} | Real worm')
    axs[1, 1].hist(agent_state_velocities, bins=20)
    axs[1, 1].set_title(f'Velocity for state {key} | Agent')

    axs[0, 2].hist(worm_durations[i], bins=20)
    axs[0, 2].set_title(f'Durations for state {key} | Real worm')
    axs[1, 2].hist(agent_durations[i], bins=20)
    axs[1, 2].set_title(f'Durations for state {key} | Agent')

    axs[0, 3].plot(range(len(worm_durations[i])), worm_durations[i], label="original")
    axs[0, 3].set_title(f'Durations for state {key} | Real worm')
    axs[1, 3].plot(range(len(agent_durations[i])), agent_durations[i])
    axs[1, 3].set_title(f'Durations for state {key} | Agent')
    xs = np.array(range(len(worm_durations[i])))

    '''if xs.size>0:
        ys = worm_durations[i]
        avg = np.mean(ys)
        std = np.std(ys)
        log_ys = np.log1p(ys)  # log1p to handle zeros safely

        # Now apply a robust method
        log_median = np.median(log_ys)
        log_mad = median_abs_deviation(log_ys)

        breaking_point = np.where(np.diff(log_ys) >= 4 * log_mad)[0][0]

        t_switch = -1

        print("avg, std=", avg, std)
        print("breaking point=", breaking_point)
        #p = np.polyfit(xs, ys, 1)
        #poly1d_fn1 = np.polyval(coefficients[0], xs[:min_j])
        if avg<std:
            axs[0,3].axvline(x=breaking_point, color='b')'''

    plt.legend()
    plt.savefig(f"states/{frame_folder}/state_{key}_v_da_duration.png")

    plt.show()



fig, axs = plt.subplots(1, 2, figsize=(12, 6))
axs[0].imshow(worm_state_transitions)
plt.title(f'State transitions')
axs[0].set_title("Real worm")
axs[1].imshow(agent_state_transitions)
axs[1].set_title("Agent")
axs[0].set_xticks(range(len(state_dic)), state_dic.keys(), rotation=45)
axs[1].set_xticks(range(len(state_dic)), state_dic.keys(), rotation=45)
axs[0].set_yticks(range(len(state_dic)), state_dic.keys(), rotation=45)
axs[1].set_yticks(range(len(state_dic)), state_dic.keys(), rotation=45)
plt.savefig(f"states/{frame_folder}/state_transitions.png")

plt.show()

worm_xs = range(0, len(worm_states) // (60 * 1) + 1, 120 // (60 * 1))
agent_xs = range(0, len(agent_states) // (60 * 1) + 1, 120 // (60 * 1))

state_colors = ["red", "green", "blue", "magenta", "black", "orange"]
fig, axs = plt.subplots(1, 2, figsize=(8, 4))
for i, key in enumerate(state_dic.keys()):
    worm_ys = worm_events_per_time_window[i]
    agent_ys = agent_events_per_time_window[i]
    axs[0].plot(worm_xs, worm_ys, label=key, c=state_colors[i])
    axs[0].set_title("Events per 120s window | Real worm")
    axs[1].plot(agent_xs, agent_ys, label=key, c=state_colors[i])
    axs[1].set_title("Events per 120s window | Agent")
    if worm_ys.size>0:
        axs[0].plot(worm_xs, worm_transition_fitting_functions[i](worm_xs), marker="^", c=state_colors[i])
    if agent_ys.size>0:
        axs[1].plot(agent_xs, agent_transition_fitting_functions[i](agent_xs), marker="^", c=state_colors[i])

plt.legend()
plt.savefig(f"states/{frame_folder}/events_per_tau_window.png")

plt.show()


