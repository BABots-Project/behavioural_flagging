import json
import os

import numpy as np
from matplotlib import pyplot as plt, animation, patches
from tifffile import imshow

from worm import state_dic


def load_and_animate_agents_and_grid2(json_file_path, fps, dest_file_path="animation.mp4"):
    # Load JSON data for agents
    with open(json_file_path, 'r') as f:
        data = json.load(f)

    # Extract parameters
    parameters = data['parameters']
    N = parameters['N']
    LOGGING_INTERVAL = parameters['LOGGING_INTERVAL']
    N_STEPS = parameters['N_STEPS']
    WIDTH = parameters['WIDTH']
    HEIGHT = parameters['HEIGHT']
    print(parameters)

    sub_states = [[data["sub_states"][agent][timestep] for timestep in range(int(N_STEPS // LOGGING_INTERVAL))] for agent in range(N)]
    sub_states_map = {
        0: "omega",
        1: "reversal",
        2: "pause",
        3: "line",
        4: "arc",
        5: "loop"
    }

    # Prepare the figure and axis
    fig, ax = plt.subplots()
    ax.set_xlim(0, WIDTH)
    ax.set_ylim(0, HEIGHT)

    # Create a list of scatter plot objects for each agent
    scatters = [ax.plot([], [], 'o', color='lightgreen', markersize=1.0)[0] for _ in range(N)]
    traces = [ax.plot([], [], '-', color='lightgreen', linewidth=0.5)[0] for _ in range(N)]
    position_matrix = [[data["positions"][agent][timestep] for timestep in range(int(N_STEPS // LOGGING_INTERVAL))] for agent in range(N)]
    # Parameters for the evolving Gaussian density function
    MU_X = parameters['MU_X']
    MU_Y = parameters['MU_Y']
    A = parameters['A']
    GAMMA = parameters['GAMMA']
    SIGMA_X = parameters['SIGMA_X']
    SIGMA_Y = parameters['SIGMA_Y']
    DIFFUSION_CONSTANT = parameters['DIFFUSION_CONSTANT']
    ATTRACTION_STRENGTH = parameters['ATTRACTION_STRENGTH']
    ATTRACTION_SCALE = parameters['ATTRACTION_SCALE']
    MAX_CONCENTRATION = parameters['MAX_CONCENTRATION']
    if MAX_CONCENTRATION>0:
        ax.text(MU_X, MU_Y, "X", color="red", fontsize=20, fontweight="bold", ha='center', va='center')


    # Function to calculate the evolving Gaussian density
    def calculate_gaussian_density(t, X, Y):
        dx = X - MU_X
        dy = Y - MU_Y
        a_t = A * np.exp(-GAMMA * t)
        sigma_x_t = SIGMA_X + 2 * DIFFUSION_CONSTANT * t
        sigma_y_t = SIGMA_Y + 2 * DIFFUSION_CONSTANT * t
        density = MAX_CONCENTRATION * a_t * np.exp(-0.5 * ((dx * dx) / (sigma_x_t * sigma_x_t) + (dy * dy) / (sigma_y_t * sigma_y_t)))
        return ATTRACTION_STRENGTH * np.log(density + ATTRACTION_SCALE)

    # Create a grid of (x, y) coordinates
    x = np.linspace(0, WIDTH, N)
    y = np.linspace(0, HEIGHT, N)
    X, Y = np.meshgrid(x, y)

    # Calculate initial Gaussian density
    Z = calculate_gaussian_density(0, X, Y)
    im = ax.imshow(Z, extent=[0, WIDTH, 0, HEIGHT], origin='lower', cmap='Purples')
    # Add colorbar
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label('Chemical Concentration')

    # Initialization function to set up the scatter plot and grid
    def init():
        for i, scatter in enumerate(scatters):
            scatter.set_data([position_matrix[i][0][0]], [position_matrix[i][0][1]])
        im.set_data(calculate_gaussian_density(0, X, Y))
        #ax.set_title(sub_states_map[sub_states[0][0]])
        if MAX_CONCENTRATION>0:
                ax.text(MU_X, MU_Y, "X", color="black", fontsize=20, fontweight="bold", ha='center', va='center')
        return scatters + [im]

    # Animation update function
    def update(frame):
        print(frame)
        for i, (scatter, trace) in enumerate(zip(scatters, traces)):
            #ax.text(MU_X, MU_Y, "X", color="red", fontsize=20, fontweight="bold", ha='center', va='center')
            scatter.set_data([position_matrix[i][frame][0]], [position_matrix[i][frame][1]])

            # Handle trace with periodic boundary conditions
            trace_x = []
            trace_y = []
            start_frame = max(0, frame - 20)  # Limit trace to 20 frames
            for j in range(start_frame, frame + 1):
                x_prev, y_prev = position_matrix[i][j - 1] if j > 0 else position_matrix[i][j]
                x_curr, y_curr = position_matrix[i][j]

                # Check for boundary crossings and adjust coordinates
                if abs(x_curr - x_prev) >= WIDTH / 2 or abs(y_curr - y_prev) >= HEIGHT / 2:
                    break

                trace_x.append(x_curr)
                trace_y.append(y_curr)

            trace.set_data(trace_x, trace_y)

        # Update the Gaussian density function
        Z = calculate_gaussian_density(frame, X, Y)
        im.set_data(Z)
        #ax.set_title(sub_states_map[sub_states[0][frame]])
        return scatters + traces + [im]

    # Create the animation
    anim = animation.FuncAnimation(
        fig, update, init_func=init, frames=int(N_STEPS // LOGGING_INTERVAL), blit=False
    )
    anim.save(
        dest_file_path + f"N_{N}_LOGGING_INTERVAL_{LOGGING_INTERVAL}_N_STEPS_{N_STEPS}.mp4",
        writer='ffmpeg', fps=fps
    )


def load_and_animate_heatmaps(heatmap_folder_path, single_file_name, json_file_path):
    # Load JSON data for parameters
    fig, ax = plt.subplots()

    with open(json_file_path, 'r') as f:
        data = json.load(f)
        parameters = data['parameters']


    N_STEPS = parameters['N_STEPS']
    LOGGING_INTERVAL = parameters['LOGGING_INTERVAL']
    WIDTH = parameters['WIDTH']
    HEIGHT = parameters['HEIGHT']
    N = parameters['N']
    DX = WIDTH/256


    scatters = [ax.plot([], [], 'o', color='black', markersize=1.0)[0] for _ in range(N)]
    traces = [ax.plot([], [], '-', color='black', linewidth=0.5)[0] for _ in range(N)]
    position_matrix = [[data["positions"][agent][timestep] for timestep in range(int(N_STEPS // LOGGING_INTERVAL))] for agent in range(N)]
    position_matrix = [[[pos[1], pos[0]] for pos in agent_pos]
                       for agent_pos in position_matrix]
    # Parse heatmap data from .txt files
    timesteps = N_STEPS // LOGGING_INTERVAL
    list_of_matrices = []
    for t in range(0, N_STEPS, LOGGING_INTERVAL):
        file_path = os.path.join(heatmap_folder_path, f'{single_file_name}_{t}.txt')
        with open(file_path, 'r') as f:
            matrix = np.loadtxt(f)
            list_of_matrices.append(matrix)

    heatmaps = list_of_matrices.copy()

    # Initialize the plot
    im = ax.imshow(heatmaps[0], extent=[0, WIDTH, 0, HEIGHT], origin='lower', cmap='viridis')
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label('Repulsive pheromone concentration')
    centers = [[43.000000, 30.000000], [19.482779, 37.641209], [34.017223, 17.636265], [34.017220, 42.363735],
               [19.482780, 22.358788]]
    #centers = [[183, 128], [83, 160], [145, 75], [145, 180], [ 83, 95]]
    centers = [[c[1] - 5*DX, c[0] - 5*DX] for c in centers]
    print("width, N=", WIDTH, N)
    print("DX: ", DX)
    for c in centers:
        print("center at: ", c)
        rect = patches.Rectangle((c[0], c[1]), 10*DX , 10*DX, linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
    # Initialization function to set up the heatmap
    def init():
        for i, scatter in enumerate(scatters):
            scatter.set_data([position_matrix[i][0][0]], [position_matrix[i][0][1]])
        im.set_data(heatmaps[0])
        return scatters + [im]

    # Animation update function
    def update(frame):
        print(frame)
        for i, (scatter, trace) in enumerate(zip(scatters, traces)):
            # ax.text(MU_X, MU_Y, "X", color="red", fontsize=20, fontweight="bold", ha='center', va='center')
            scatter.set_data([position_matrix[i][frame][0]], [position_matrix[i][frame][1]])

            # Handle trace with periodic boundary conditions
            trace_x = []
            trace_y = []
            start_frame = max(0, frame - 20)  # Limit trace to 20 frames
            for j in range(start_frame, frame + 1):
                x_prev, y_prev = position_matrix[i][j - 1] if j > 0 else position_matrix[i][j]
                x_curr, y_curr = position_matrix[i][j]

                # Check for boundary crossings and adjust coordinates
                if abs(x_curr - x_prev) >= WIDTH / 2 or abs(y_curr - y_prev) >= HEIGHT / 2:
                    break

                trace_x.append(x_curr)
                trace_y.append(y_curr)

            trace.set_data(trace_x, trace_y)
        vmin = np.min(heatmaps[frame])
        vmax = np.max(heatmaps[frame])
        im.set_clim(vmin, vmax)
        im.set_data(heatmaps[frame])
        return [im] + scatters + traces

    # Create the animation
    anim = animation.FuncAnimation(
        fig, update, init_func=init, frames=timesteps, blit=False
    )
    anim.save('heatmap_animation.mp4', writer='ffmpeg', fps=30)


def animate_real_worm_with_state_changes(xs, ys, states, speeds):
    """
    Animate the worm's movement frame by frame, displaying state changes.

    Parameters:
    xs (list or np.array): X coordinates of the worm's trajectory.
    ys (list or np.array): Y coordinates of the worm's trajectory.
    states (list or np.array): State values (0-5) associated with each frame.
    """
    state_colors = ["blue", "green", "red", "purple", "orange", "brown"]  # Colors for states 0-5

    plt.ion()  # Turn on interactive mode
    fig, ax = plt.subplots()
    ax.set_xlabel("X Position")
    ax.set_ylabel("Y Position")

    for i in range(len(xs)):
        ax.clear()
        ax.set_title(f"Frame {i + 1}, State: {list(state_dic.keys())[int(states[i])]}, Speed: {speeds[i]:.2f}")
        ax.set_xlabel("X Position")
        ax.set_ylabel("Y Position")

        # Plot trajectory up to the current frame
        ax.plot(xs[:i + 1], ys[:i + 1], color="gray", linestyle="--", alpha=0.5)

        # Plot previous positions with reduced visibility
        ax.scatter(xs[:i], ys[:i], color="gray", alpha=0.3, s=50)

        # Highlight the current position with its state color
        ax.scatter(xs[i], ys[i], color=state_colors[int(states[i])], s=100, label=f"State {list(state_dic.keys())[int(states[i])]}")
        ax.legend()

        plt.draw()
        plt.pause(0.1)  # Small pause to render the frame
        input("Press Enter to continue...")  # Wait for user input before moving to the next frame

    plt.ioff()  # Turn off interactive mode
    plt.show()


# Main execution
if __name__ == "__main__":
    base_dir = "/home/nema/cuda_agent_based_sim/"
    #load_and_animate_agents_and_grid2(base_dir + "agents_all_data.json", fps=30, dest_file_path=base_dir)
    load_and_animate_heatmaps(base_dir+"logs/repulsive_pheromone/", "repulsive_pheromone_step", base_dir + "agents_all_data.json")
    #load_and_animate_heatmaps(base_dir+"logs/bacterial_lawn/", "bacterial_lawn_step", base_dir + "agents_all_data.json")