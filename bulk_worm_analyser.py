from traceback import print_tb

import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import beta, vonmises, gamma

from main_analyzer import calculate_angles, calculate_angle_changes
from runner import zenodo_frame_folders
from worm import state_dic


def load_data():
    state_velocities = []
    state_angle_changes = []

    subsampling_rate = 1  # changing this requires changing the indexing of angle changes wrt states (idk how, so just leave it at that)
    durations = {}
    per_worm_durations = [{} for _ in range(len(zenodo_frame_folders))]

    for i, key in enumerate(state_dic.keys()):
        durations[i] = []

    for i in range(len(zenodo_frame_folders)):
        for j in range(len(state_dic.keys())):
            per_worm_durations[i][j]=[]

    max_t = 0
    time_window = 120 * 4
    all_velocities = []  # 1 per each worm
    all_angle_changes = []  # ''
    all_durations = []  # 1 list of durations for each worm
    all_duration_indices = []
    all_states = []


    for folder_id, frame_folder in enumerate(zenodo_frame_folders):
        worm_velocities = np.load(f"states/{frame_folder}/velocities.npy")[::subsampling_rate]
        # find outliers (above 300), they will be eradicated
        outlier_indices = np.where(worm_velocities > 600)[0]
        outlier_mask = np.ones(len(worm_velocities), dtype=bool)
        outlier_mask[outlier_indices] = False
        worm_states = np.load(f"states/{frame_folder}/states.npy")[::subsampling_rate][outlier_mask]
        worm_positions = np.load(f"states/{frame_folder}/positions.npy")
        xs = np.array([x for x in worm_positions[0]][::subsampling_rate])[outlier_mask]
        ys = np.array([x for x in worm_positions[1]][::subsampling_rate])[outlier_mask]
        worm_positions = [xs, ys]
        worm_velocities = worm_velocities[outlier_mask]
        worm_angles = calculate_angles(worm_positions[0], worm_positions[1])
        worm_angle_changes = calculate_angle_changes(worm_angles)
        worm_angle_changes = [0.0, 0.0] + worm_angle_changes
        worm_angle_changes = np.array(worm_angle_changes)
        all_velocities.append(worm_velocities)
        all_angle_changes.append(worm_angle_changes)
        all_states.append(worm_states)

        for i, key in enumerate(state_dic.keys()):
            if len(state_velocities)<=i:
                state_velocities.append([])
            if len(state_angle_changes)<=i:
                state_angle_changes.append([])


            worm_state_indices = np.where(worm_states == state_dic[key])[0]
            worm_state_indices = worm_state_indices[np.where(worm_state_indices < len(worm_velocities))]

            worm_state_angle_changes = worm_angle_changes[worm_state_indices]
            worm_state_velocities = np.array(worm_velocities)[worm_state_indices]

            state_velocities[i].append(worm_state_velocities)
            state_angle_changes[i].append(worm_state_angle_changes)

        if len(worm_states) > max_t:
            max_t = len(worm_states)

        state_transitions = np.zeros((len(state_dic), len(state_dic)))
        current_duration = 0
        worm_durations = []
        worm_duration_indices = []

        for t in range(len(worm_states) - 1):
            state = int(worm_states[t])
            next_state = int(worm_states[t + 1])
            if state == next_state:
                current_duration += 1
            else:
                if current_duration != 0 and state != -1:
                    durations[state].append(current_duration)
                    per_worm_durations[folder_id][state].append(current_duration)
                    worm_durations.append(current_duration)
                    worm_duration_indices.append(t)
                current_duration = 0
                state_transitions[state][next_state] += 1
        all_durations.append(worm_durations)
        all_duration_indices.append(worm_duration_indices)

    return state_velocities, state_angle_changes, durations, all_velocities, all_states, all_durations, all_duration_indices, all_angle_changes, per_worm_durations



def get_likelihood(state_velocities, state_angle_changes, durations, all_velocities, all_states, all_durations, all_duration_indices,  all_angle_changes, per_worm_durations, plot_histograms = False):
    rng = np.random.default_rng()

    test_indices = np.random.choice(len(zenodo_frame_folders), int(0.3 * len(zenodo_frame_folders)), replace=False)
    train_indices = np.setdiff1d(list(range(len(zenodo_frame_folders))), test_indices)
    #print(state_velocities)
    #print("test: ", test_indices)
    train_speed = [[] for _ in state_velocities]
    test_speed = [[] for _ in state_velocities]
    train_angle_change = [[] for _ in state_angle_changes]
    test_angle_change = [[] for _ in state_angle_changes]
    train_durations = [[] for _ in durations]
    test_durations = [[] for _ in durations]
    for i in range(len(state_velocities)):
        state_vs = state_velocities[i]
        state_as = state_angle_changes[i]

        for j in range(len(state_vs)):
            if j in train_indices:
                train_speed[i].extend(state_vs[j])
                train_angle_change[i].extend(state_as[j])
                train_durations[i].extend(per_worm_durations[j][i])
            else:
                test_speed[i].extend(state_vs[j])
                test_angle_change[i].extend(state_as[j])
                test_durations[i].extend(per_worm_durations[j][i])


    #print(sensible_events_per_time_window)
    fitted_params = {
        "speed" : {
            "v_min" : -1,
            "scale" : -1,
            "a" : -1,
            "b" : -1
        },
        "speed_gamma": {
            "v_min": -1,
            "scale": -1,
            "a": -1
        },
        "angle_change" : {
            "mu" : -1,
            "kappa" : -1,
        },
        "duration":{
            "d_min" : -1,
            "scale" : -1,
            "a" : -1,
            "b" : -1
        },
        "duration_gamma":{
            "d_min" : -1,
            "scale" : -1,
            "a": -1
        }
    }

    state_fitted_params = [fitted_params for i in range(len(list(state_dic)))]

    for i, key in enumerate(state_dic.keys()):


        data_vel = np.array(train_speed[i])
        vel_min, vel_max = data_vel.min(), data_vel.max()
        epsilon = 1e-6
        vel_norm = np.clip((data_vel - vel_min) / (vel_max - vel_min), epsilon, 1 - epsilon)
        try:
            a_vel, loc_vel, scale_vel = gamma.fit(vel_norm, floc=0, fscale=1)
            state_fitted_params[i]["speed_gamma"]["v_min"] = vel_min
            state_fitted_params[i]["speed_gamma"]["scale"] = vel_max - vel_min
            state_fitted_params[i]["speed_gamma"]["a"] = a_vel
            #state_fitted_params[i]["speed"]["b"] = b_vel
        except:
            print("Error in speed fitting:")
            print("speed: ", data_vel)
            print("speed norm: ", vel_norm)
            print("max, min: ", vel_max, vel_min)
            raise ValueError


        p = vonmises.fit(train_angle_change[i])
        #print("p: ", p)
        kappa, mu, _ = p
        state_fitted_params[i]["angle_change"]["mu"] = mu
        state_fitted_params[i]["angle_change"]["kappa"] = kappa


        data_dur = np.array(train_durations[i])
        if data_dur.size>0:
            dur_min, dur_max = data_dur.min(), data_dur.max()
            if dur_min!=dur_max:
                #dur_norm = (data_dur - dur_min) / (dur_max - dur_min)
                dur_norm = np.clip((data_dur - dur_min)/(dur_max - dur_min), epsilon, 1-epsilon)
                try:
                    '''a_dur, b_dur, loc_dur, scale_dur = beta.fit(dur_norm, floc=0, fscale=1)
                    state_fitted_params[i]["duration"]["d_min"] = dur_min
                    state_fitted_params[i]["duration"]["scale"] = dur_max - dur_min
                    state_fitted_params[i]["duration"]["a"] = a_dur
                    state_fitted_params[i]["duration"]["b"] = b_dur'''
                    a_dur, loc_dur, scale_dur = gamma.fit(dur_norm, floc=0, fscale=1)
                    state_fitted_params[i]["duration_gamma"]["d_min"] = dur_min
                    state_fitted_params[i]["duration_gamma"]["scale"] = dur_max - dur_min
                    state_fitted_params[i]["duration_gamma"]["a"] = a_dur
                except:
                    print("Error in duration fitting:")
                    print("duration: ", data_dur)
                    print("duration norm: ", dur_norm)
                    print("max, min: ", dur_max, dur_min)
                    raise ValueError

        else:
            print("No duration training data!")
            print("state: ", i)
            print("train_durations: ", train_durations)
            #raise ValueError

        if plot_histograms:
            fig, axs = plt.subplots(2, 3, figsize=(24, 8))
            axs[0, 1].hist(train_speed[i], bins=20)
            axs[0, 1].set_title(f'|V| train | {key}')
            axs[1, 1].hist(test_speed[i], bins=20)
            axs[1, 1].set_title(f'|V| test | {key}')
            axs[0, 0].hist(train_angle_change[i], bins=20)
            axs[0, 0].set_title(f'd-angle train | {key}')
            axs[1, 0].hist(test_angle_change[i], bins=20)
            axs[1, 0].set_title(f'd-angle test | {key}')
            axs[0, 2].hist(train_durations[i], bins=20)
            axs[0, 2].set_title(f'durations train | {key}')
            axs[1, 2].hist(test_durations[i], bins=20)
            axs[1, 2].set_title(f'durations test | {key}')
            plt.show()


    def calculate_likelihood(speed, angle, duration, state_idx, fitted_params):
        # Retrieve parameters for the given state
        params = fitted_params[state_idx]

        # --- Speed likelihood (Beta) ---
        # Normalize using training parameters
        speed_min = params['speed_gamma']['v_min']
        speed_scale = params['speed_gamma']['scale']
        #speed_norm = (speed - speed_min) / speed_scale
        epsilon = 1e-6
        speed_norm = np.clip((speed-speed_min)/speed_scale, epsilon, 1 - epsilon)

        a_speed = params['speed_gamma']['a']
        #b_speed = params['speed']['b']
        if a_speed!=-1:# and b_speed!=-1:
            #pdf_speed = beta.pdf(speed_norm, a_speed, b_speed, loc=0, scale=1)
            pdf_speed = gamma.pdf(speed_norm, a_speed, loc=0, scale=1)

        else:
            pdf_speed = 1
        # --- Angle likelihood (Von Mises) ---
        kappa = params['angle_change']['kappa']
        loc_angle = params['angle_change']['mu']
        if kappa!=-1 and loc_angle!=-1:
            pdf_angle = vonmises.pdf(angle, kappa, loc=loc_angle)
        else:
            pdf_angle = 1

        if duration>0: #otherwise, it's not the beginning of a state segment
            # --- Duration likelihood (Beta) ---
            duration_min = params['duration_gamma']['d_min']
            duration_scale = params['duration_gamma']['scale']
            duration_norm = np.clip((duration - duration_min) / duration_scale, epsilon, 1-epsilon)
            a_duration = params['duration_gamma']['a']
            #b_duration = params['duration']['b']
            if a_duration!=-1: #and b_duration!=-1:
                pdf_duration = gamma.pdf(duration_norm, a_duration, loc=0, scale=1)
            else:
                pdf_duration = 1
        else:
            pdf_duration = 1
            duration_norm = 1
        if np.isinf(pdf_speed) or np.isinf(pdf_angle) or np.isinf(pdf_duration):
            print("found INF likelihood:")
            print("state: ", state_idx)
            print("fitted params: ", fitted_params)
            print(f'speed {speed} |speed| {speed_norm} pdf: {pdf_speed}')
            print(f'angle {angle} pdf: {pdf_angle}')
            print(f'duration {duration} |duration| {duration_norm} pdf: {pdf_duration}')
            raise ValueError
        # Combine likelihoods (multiplicative, or add logs for numerical stability)
        likelihood = pdf_speed * pdf_angle * pdf_duration
        log_likelihood = np.log(pdf_speed) + np.log(pdf_angle) + np.log(pdf_duration)

        return likelihood, log_likelihood

    likelihoods = []
    log_likelihoods = []

    flat_likelihoods = []
    flat_lls = []

    for idx in test_indices:
        vs = all_velocities[idx]
        s = all_states[idx]
        ac = all_angle_changes[idx]
        ds = all_durations[idx]
        d_ids = all_duration_indices[idx]
        cur_duration_idx = 0
        cur_l = []
        cur_ll = []
        for i, state in enumerate(s):
            d=-1
            if i in d_ids:
                d = ds[cur_duration_idx]
                #print("duration: ", d)
                cur_duration_idx+=1

            l, ll = calculate_likelihood(vs[i], ac[i], d, int(s[i]), state_fitted_params)
            cur_l.append(l)
            cur_ll.append(ll)
            flat_likelihoods.append(l)
            flat_lls.append(ll)
        #print(f'index {idx} has mean likelihood: {np.mean(cur_l)} and mean log-likelihood {np.mean(cur_ll)} ')
        likelihoods.append(cur_l)
        log_likelihoods.append(cur_ll)

    '''print("likelihoods: ", likelihoods)
    print("log-likehoods: ", log_likelihoods)'''
    #print("mean likelihood: ", np.mean(flat_likelihoods))
    #print("mean log-likelihood: ", np.mean(flat_lls))
    return np.mean(flat_likelihoods), np.mean(flat_lls)

n_iter = 1000
likelihoods = []
log_likelihoods = []
state_velocities, state_angle_changes, durations, all_velocities, all_states, all_durations, all_duration_indices, all_angle_changes, per_worm_durations = load_data()
for i in range(n_iter):
    likelihood, log_likelihood = get_likelihood(state_velocities, state_angle_changes, durations, all_velocities, all_states, all_durations, all_duration_indices, all_angle_changes, per_worm_durations)
    print(f'iter {i} l={likelihood} ll={log_likelihood}')
    likelihoods.append(likelihood)
    log_likelihoods.append(log_likelihood)

plt.hist(likelihoods, bins=20)
plt.show()
plt.close()
plt.hist(log_likelihoods, bins=20)
plt.show()
print(f'likelihood max, min, mean = {np.max(likelihoods), np.min(likelihoods), np.mean(likelihoods)}')
print(f'log-likelihood max, min, mean = {np.max(log_likelihoods), np.min(log_likelihoods), np.mean(log_likelihoods)}')

