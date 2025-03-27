import numpy as np
import scipy as sp
from matplotlib import pyplot as plt


def predict(worm_states, beta_as, beta_bs, beta_scales, mus, kappas, duration_as, duration_bs, duration_scale, worm_positions, worm_msd, DESIRED_FPS):
    estimated_msd = []
    estimated_msd_x = []
    estimated_msd_y = []
    variance_msd_x = []
    variance_msd_y = []

    for i in range(1, len(worm_states)):
        state = int(worm_states[i])
        expected_velocity = beta_as[state] / (beta_bs[state] + beta_as[state]) * beta_scales[state]
        velocity_variance = beta_as[state] * beta_bs[state] / (
                    (beta_as[state] + beta_bs) ** 2 * (beta_as[state] + beta_bs[state] + 1)) * beta_scales[state]
        expected_angle = mus[state]
        angle_variance = sp.i1(kappas[state]) / sp.i0(kappas[state])
        expected_duration = duration_as[state] / (duration_bs[state] + duration_as[state]) * duration_scale[state]

        if expected_duration < 0: expected_duration = 1
        msd = expected_velocity  # * (np.pi - expected_angle)
        # if state == state_dic["reversal"]:  msd = -msd
        estimated_msd.append(msd)
        v_x = expected_velocity * np.cos(expected_angle)
        v_y = expected_velocity * np.sin(expected_angle)
        estimated_msd_x.append(v_x)
        estimated_msd_y.append(v_y)
        variance_msd_x.append(((velocity_variance ** 2 + expected_velocity ** 2) * (
                    np.cos(expected_angle) ** 2 + np.cos(angle_variance) ** 2) - expected_velocity ** 2 * np.cos(
            expected_angle) ** 2))
        variance_msd_y.append(((velocity_variance ** 2 + expected_velocity ** 2) * (
                    np.sin(expected_angle) ** 2 + np.sin(angle_variance) ** 2) - expected_velocity ** 2 * np.sin(
            expected_angle) ** 2))

    estimated_msd = np.cumsum(estimated_msd)
    estimated_msd = np.pow(estimated_msd, 2) * DESIRED_FPS

    estimated_msd_x = np.pow(np.cumsum(estimated_msd_x), 2)
    estimated_msd_y = np.pow(np.cumsum(estimated_msd_y), 2)
    variance_msd_x = np.pow(np.cumsum(variance_msd_x), 2)
    variance_msd_y = np.pow(np.cumsum(variance_msd_y), 2)

    estimated_msd_decomposed = estimated_msd_x + estimated_msd_y
    variance_msd_decomposed = variance_msd_x + variance_msd_y

    plt.plot(range(1, len(worm_positions[0])), np.log10(worm_msd), label="actual worm")
    plt.plot(range(1, len(worm_positions[0])), np.log10(estimated_msd), label="estimation")
    plt.plot(range(1, len(worm_positions[0])), np.log10(estimated_msd_decomposed), label="estimation decomposed")
    plt.plot(range(1, len(worm_positions[0])), np.log10(variance_msd_decomposed), label="variance decomposed")

    plt.title("MSD")
    plt.legend()
    plt.show()
    plt.close()