import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import vonmises, beta, median_abs_deviation
import json

#from runner import DESIRED_FPS
from worm import state_dic
import scipy.special as sp


def calculate_angles(x, y):
    angles = []
    for i in range(1, len(x)):
        dx = x[i] - x[i - 1]
        dy = y[i] - y[i - 1]
        angle = np.arctan2(dy, dx)
        angles.append(angle)
    return angles


def calculate_angle_changes(angles):

    angle_changes_ = np.diff(angles) # Changes in angle between consecutive frames
    angle_changes_ = [angle_changes_[i] - 2 * np.pi if angle_changes_[i] > np.pi else angle_changes_[i] for i in range(len(angle_changes_))]
    angle_changes_ = [angle_changes_[i] + 2 * np.pi if angle_changes_[i] < -np.pi else angle_changes_[i] for i in range(len(angle_changes_))]

    return angle_changes_


def analyze(worm_states, worm_positions, fit_distributions=False, angles=[], velocities=[], log_state_data=False, target="Real Worm", fps=4):
    DESIRED_FPS = fps

    time_window = 120 * DESIRED_FPS
    if len(angles)<1:
        angles = calculate_angles(worm_positions[0], worm_positions[1])

    angle_changes =calculate_angle_changes(angles)
    angle_changes = np.append(np.array(0.0), angle_changes)
    angle_changes = np.append(angle_changes, np.array(0.0))

    if len(velocities)<1:
        positions = worm_positions.T
        velocities = np.append(np.array(0.0), np.abs(np.diff(positions, axis=0)))
    '''else:
        velocities = np.roll(velocities, 1)
        velocities[0]=0.0'''

    kappas = []
    mus = []
    beta_as = []
    beta_bs = []
    beta_scales = []
    max_state_speed=np.zeros(len(state_dic)) - 1
    min_state_speed = np.zeros(len(state_dic)) - 1
    durations = {}
    #fig, axs = plt.subplots(len(state_dic), 2, figsize=(24, 8*len(state_dic)))
    for i, key in enumerate(state_dic.keys()):
        #fig, axs = plt.subplots(1, 2, figsize=(12, 6))

        durations[i]=[]
        state_indices = np.where(worm_states==state_dic[key])[0]
        state_angle_changes = angle_changes[state_indices]
        state_velocities = velocities[state_indices]

        state_velocities = state_velocities[np.where(state_velocities<=300)]
        if state_velocities.size>0:
            max_state_speed[i]=np.max(state_velocities)
            min_state_speed[i]=np.min(state_velocities)
        a = -1
        b = -1
        loc = -1
        scale = -1
        mu = -1
        kappa = -1
        if fit_distributions and state_angle_changes.size>0:
            kappa, mu, scale = vonmises.fit(state_angle_changes)
            #print(f'state {key} has')
            #print(f'\t angle parameters kappa={round(kappa,2)}, loc={round(loc, 2)}, scale={scale}, variance={1-sp.i1(kappa)/sp.i0(kappa)}')

            #sample = vonmises(loc=loc, kappa=kappa).rvs(len(state_indices))
            #axs[0].hist(sample, bins=20, alpha=0.2, color="red")
            if state_velocities.size > 0:
                params = beta.fit(state_velocities)

                a, b, loc, scale = params


            #print(
               # f'\t velocity parameters a={round(a, 2)}, b={round(b, 2)}, scale={round(scale)} E(V)={round(a / (a + b) * scale, 2)} ST(V)={round(a * b / ((a + b) ** 2 + (a + b + 1)) * scale, 2)}')
        beta_as.append(a)
        beta_bs.append(b)
        beta_scales.append(scale)
        kappas.append(kappa)
        mus.append(mu)
            #velocity_samples = beta(a=a, b=b, loc=loc, scale=scale).rvs(len(state_indices))
            #axs[1].hist(velocity_samples, bins=20, alpha=0.2, color="red")

        #axs[0].hist(state_angle_changes, bins=20)
        #axs[0].set_title(f'Angle changes for state {key} | {target}')

        #axs[1].hist(state_velocities, bins=20)
        #axs[1].set_title(f'Velocity for state {key} | {target}')

        #plt.show()

    #plt.close()



    state_transitions = np.zeros((len(state_dic), len(state_dic)))
    current_duration = 0
    for t in range(len(worm_states)-1):
        state = int(worm_states[t])
        next_state = int(worm_states[t+1])
        if state==next_state:
            current_duration+=1
        else:
            if current_duration!=0:
                durations[state].append(current_duration)
            current_duration=0
            state_transitions[state][next_state]+=1

    times = range(0, len(worm_states), time_window)
    time_state_transitions = np.zeros((int(len(worm_states)/time_window)+1, len(state_dic), len(state_dic)))

    for i, t in enumerate(times):
        for j in range(t, t+time_window-1):
            if j<len(worm_states):
                state = int(worm_states[j])
            if j+1<len(worm_states):
                next_state = int(worm_states[j+1])
                if state!=next_state:
                    time_state_transitions[i][state][next_state]+=1

    for i, t in enumerate(time_state_transitions):
        for j, row in enumerate(t):
            sum_ = sum(row)
            if sum_!=0:
                t[j]/=sum_

    durations_matrix = []
    duration_as = []
    duration_bs = []
    duration_scale = []
    plt.close()
    #fig, axs = plt.subplots(len(state_dic.keys()), 1, figsize=(8, 8*len(state_dic.keys())))
    t_breaking_points = np.zeros(len(state_dic.keys())) - 1
    state_duration_scales1 = np.zeros(len(state_dic.keys())) - 1
    state_duration_scales2 = np.zeros(len(state_dic.keys())) - 1
    for i, row in enumerate(state_transitions):
        #fig, axs = plt.subplots(1, 1, figsize=(6, 6))

        sum_ = sum(row)
        if sum_>0:
            state_transitions[i] /= sum_
        else:
            state_transitions[i] = 0.0

        durations_matrix.append(durations[i])
        if len(durations[i])>0:
            if fit_distributions:
                loc = 1
                p = beta.fit(durations[i], floc=loc)
                a, b, loc, scale = p
                while a>10**3 or b>10**3 or scale>10**5:
                    loc+=1
                    p = beta.fit(durations[i], floc=loc)
                    a, b, loc, scale = p
                    if loc>100:
                        a=-1
                        b=-1
                        scale=-1
                        break

                state_indices = np.where(worm_states == i)[0]
                ys = durations[i]
                avg = np.mean(ys)
                std = np.std(ys)
                t_breaking_point = -1
                scale1 = scale
                scale2 = scale
                print(f'state {i}')
                print(f'avg {avg} std {std}')
                if len(ys)>0:
                    log_ys = np.log1p(ys) #ys  # log1p to handle zeros safely
                    log_mad = median_abs_deviation(log_ys)
                    if abs(log_mad)>0:
                        n_mads = 3
                        print("mad = ", log_mad)
                        spiking_indices = np.where(np.diff(log_ys) >= n_mads * log_mad)[0]
                        if spiking_indices.size>0:
                            print("breaking points: ")
                            print(spiking_indices)

                            breaking_point = spiking_indices[0] #take first time it happens
                            sum_of_durations = int(np.sum(durations[i][:breaking_point]))
                            print("sum of durations: ")
                            print(sum_of_durations)
                            t_breaking_point = state_indices[sum_of_durations]
                            if breaking_point==0:   breaking_point=1
                            scale1 = np.max(durations[i][:breaking_point])
                            scale2 = np.max(durations[i])

                duration_as.append(a)
                duration_bs.append(b)
                duration_scale.append(scale)
                t_breaking_points[i] = t_breaking_point
                state_duration_scales1[i] = scale1
                state_duration_scales2[i] = scale2

        else:
            duration_as.append(-1)
            duration_bs.append(-1)
            duration_scale.append(-1)
        '''if fit_distributions:
            print(f'state {i} has duration alpha = {duration_as[i]}, beta={duration_bs[i]}, scale = {duration_scale[i]}')
            print(f'original sample: {durations[i]}')'''
        #plt.show()
    #plt.imshow(state_transitions)
    #plt.title(target)
    #plt.xticks(range(len(state_dic)), state_dic.keys(), rotation=45)
    #plt.yticks(range(len(state_dic)), state_dic.keys(), rotation=45)
    #plt.show()

    events_per_time_window = np.zeros((len(state_dic.keys()), int(len(worm_states)/time_window)+1))
    i=0
    for t in range(0, len(worm_states), time_window):
        last_found = -1
        for k in range(t,min(len(worm_states), t+time_window)):
            if worm_states[k]!=last_found:
                events_per_time_window[int(worm_states[k]), i]+=1
            last_found = worm_states[k]

        i+=1

    xs = range(0, len(worm_states)//(60*DESIRED_FPS)+1, time_window//(60*DESIRED_FPS))
    normalised_events_per_time_window = np.zeros((len(state_dic.keys()), int(len(worm_states)/time_window)+1))

    for i in range(len(xs)):
        total_events_per_time_window = np.sum(events_per_time_window[:, i])
        if total_events_per_time_window>0:
            normalised_events_per_time_window[:, i] = events_per_time_window[:, i] / total_events_per_time_window


    #plt.close()
    state_colors = ["red", "green", "blue", "magenta", "black", "orange"]
    ms = []
    qs = []
    transition_fitting_functions = []
    for i, key in enumerate(state_dic.keys()):
        ys = events_per_time_window[i]
        #plt.plot(xs, ys, label=key, c=state_colors[i])
        #print(f'state {key} has ys={ys}')
        if ys.size>0:
            coef = np.polyfit(xs, ys, 1)
            poly1d_fn = np.poly1d(coef)
            transition_fitting_functions.append(poly1d_fn)
            qs.append(coef[1])
            ms.append(coef[0])
        else:
            qs.append(-1)
            ms.append(-1)
            transition_fitting_functions.append(-1)

        #plt.plot( xs, poly1d_fn(xs), marker='^', c=state_colors[i])

    #plt.legend()
    #plt.title(f"Events per {time_window // DESIRED_FPS}s window | {target}")
    #plt.show()

    #plt.close()
    for i, key in enumerate(state_dic.keys()):
        ys = normalised_events_per_time_window[i]
        coef = np.polyfit(xs, ys, 1)
        poly1d_fn = np.poly1d(coef)
        plt.plot( xs, poly1d_fn(xs), marker='^' , c=state_colors[i])
        plt.plot(xs, ys, label=key, c=state_colors[i])

    #plt.legend()
    #plt.title(f"Normalized events per {time_window // DESIRED_FPS}s window | {target}")

    #plt.show()

    transition_ms = np.zeros((len(state_dic.keys()), len(state_dic.keys())))
    transition_qs = np.zeros((len(state_dic.keys()), len(state_dic.keys())))
    time_transition_fitting_functions = []
    for i,key1 in enumerate(state_dic.keys()):
        line = []
        for j, key2 in enumerate(state_dic.keys()):
            ys = time_state_transitions[:, i, j]
            #plt.plot(xs, ys, label=key2, c=state_colors[j])
            coef = np.polyfit(xs, ys, 1)
            poly1d_fn = np.poly1d(coef)
            line.append(poly1d_fn)
            transition_ms[i,j] = coef[1]
            transition_qs[i,j] = coef[0]
            #plt.plot(xs, poly1d_fn(xs), marker='^', c=state_colors[j])
        time_transition_fitting_functions.append(line)
        #plt.title(f"Transitions from state {key1} per {time_window // DESIRED_FPS}s window | {target}")
        #plt.legend()
        #plt.show()
    plt.close()
    if log_state_data:
        state_properties_dic = {}
        for i, key in enumerate(state_dic.keys()):
            state_properties_dic[key] = {}
            state = state_properties_dic[key]
            state["angle"] = {}
            state["angle"]["mu"] = mus[i]
            state["angle"]["kappa"] = kappas[i]
            state["speed"] = {}
            state["speed"]["alpha"] = beta_as[i]
            state["speed"]["beta"] = beta_bs[i]
            state["speed"]["scale"] = beta_scales[i]
            state["speed"]["max_value"] = max_state_speed[i]
            state["speed"]["min_value"] = min_state_speed[i]
            state["probability"] = {}
            state["probability"]["m"] = ms[i]
            state["probability"]["q"] = qs[i]
            state["duration"] = {}
            state["duration"]["alpha"] = duration_as[i]
            state["duration"]["beta"] = duration_bs[i]
            state["duration"]["scale"] = duration_scale[i]
            state["duration"]["breaking_point"] = t_breaking_points[i]
            state["duration"]["scale1"] = state_duration_scales1[i]
            state["duration"]["scale2"] = state_duration_scales2[i]
            state["transition_likelihood"] = {}
            for j in range(len(state_transitions)):
                if i!=j:
                    if np.isnan(state_transitions[i][j]):
                        state_transitions[i][j] = 0.0
                    state["transition_likelihood"][list(state_dic.keys())[j]] = state_transitions[i][j]

        '''state_properties_dic["transition_probability"] = {}
        for i, row in enumerate(state_transitions):
            state_properties_dic["transition_probability"][list(state_dic.keys())[i]] = {}
            for j, col in enumerate(row):
                state_properties_dic["transition_probability"][list(state_dic.keys())[i]][list(state_dic.keys())[j]] = {}
                state_properties_dic["transition_probability"][list(state_dic.keys())[i]][list(state_dic.keys())[j]]["m"] = transition_ms[i,j]
                state_properties_dic["transition_probability"][list(state_dic.keys())[i]][list(state_dic.keys())[j]]["q"] = transition_qs[i,j]'''




        with open('state_estimations/worm_1/state_data.json', 'w', encoding='utf-8') as f:
            json.dump(state_properties_dic, f, ensure_ascii=False, indent=4)

    return angle_changes, durations, state_transitions, events_per_time_window, transition_fitting_functions, time_state_transitions, time_transition_fitting_functions

def analyze_agglomerated_runs(agent_states, agent_angle_changes, agent_velocities, agent_durations, worm_states, worm_angle_changes, worm_velocities, worm_durations):
    run_states = {"line": 3, "arc": 4, "loop": 5}
    run_states_indices = range(3, 6)
    fig, axs = plt.subplots(2, 4, figsize=(16, 8))
    agent_state_indices = []
    for t, s in enumerate(agent_states):
        if s in run_states_indices:
            agent_state_indices.append(t)
    agent_state_indices = np.array(agent_state_indices)
    agent_state_angle_changes = agent_angle_changes[agent_state_indices]
    agent_state_velocities = agent_velocities[agent_state_indices]

    worm_state_indices = []
    for t, s in enumerate(worm_states):
        if s in run_states_indices:
            worm_state_indices.append(t)
    worm_state_indices = np.array(worm_state_indices)
    worm_state_indices = worm_state_indices[np.where(worm_state_indices < len(worm_velocities))]
    worm_state_angle_changes = worm_angle_changes[worm_state_indices]
    worm_state_velocities = np.array(worm_velocities)[worm_state_indices]

    axs[0, 0].hist(worm_state_angle_changes, bins=20)
    axs[0, 0].set_title(f'Angle changes for state Run | Real worm')
    axs[1, 0].hist(agent_state_angle_changes, bins=20)
    axs[1, 0].set_title(f'Angle changes for state Run | Agent')

    axs[0, 1].hist(worm_state_velocities, bins=20)
    axs[0, 1].set_title(f'Velocity for state Run | Real worm')
    axs[1, 1].hist(agent_state_velocities, bins=20)
    axs[1, 1].set_title(f'Velocity for state Run | Agent')

    agent_run_durations = []
    for i in run_states.values():
        agent_run_durations.extend(agent_durations[i])
    worm_run_durations = []
    for i in run_states.values():
        worm_run_durations.extend(worm_durations[i])
    axs[0, 2].hist(worm_run_durations, bins=20)
    axs[0, 2].set_title(f'Durations for state Run | Real worm')
    axs[1, 2].hist(agent_run_durations, bins=20)
    axs[1, 2].set_title(f'Durations for state Run | Agent')

    '''axs[0, 3].plot(range(len(worm_durations[i])), worm_durations[i], label="original")
    axs[0, 3].set_title(f'Durations for state Run | Real worm')
    axs[1, 3].plot(range(len(agent_durations[i])), agent_durations[i])
    axs[1, 3].set_title(f'Durations for state {key} | Agent')'''