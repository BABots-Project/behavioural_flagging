from traceback import print_tb

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from utils import get_all_frame_paths, get_solidity, calculate_worm_size, calculate_reversals

state_dic = {"omega": 0, "reversal": 1, "pirouette": 2, "pause":3, "line":4, "arc":5, "loop":6} #0 omega, 1 reversal, 2 pirouette, 3 pause, 4 line, 5 arc, 6 loop
solidity_threshold = 0.7 # minimum solidity for an omega


class Worm:
    def __init__(self, path:str, extension:str, reversal_angle_threshold:int, scale_to_um:float, subsampling_rate:float|int, pixel_to_mm, invert_color:bool):
        '''
        :param path: folder where tracking data (pictures and centroid coordinates)
        :param extension: type of images (.jpg, .png, .tiff, etc)
        :param reversal_angle_threshold: minimum angle change for reversal detection (120 degrees)
        :param scale_to_um: stepper units to um scale (for centroid track)
        :param subsampling_rate: dividing fps by this should give 4Hz as per Salvador et al 2014
        :param pixel_to_mm: pixel density per mm (for worm pictures)
        :param invert_color: set it to true if the worm is darker than the background (for worm pictures)
        '''

        self.path = path
        self.extension = extension
        self.equivalent_diameter = -1
        self.states = []
        self.reversal_angle_threshold = reversal_angle_threshold
        self.scale_to_um = scale_to_um
        self.subsampling_rate = subsampling_rate
        self.pixel_to_mm_ratio = pixel_to_mm
        self.animal_length_um = -1.0
        self.invert_color = invert_color
        self.positional_data = []

    def load_x_y_t(self):
        worm_data = np.loadtxt(self.path + "data.txt", delimiter="\t")

        self.positional_data = pd.DataFrame({
            'Timestamp': worm_data[:, 0][::self.subsampling_rate],
            'X position': worm_data[:, 1][::self.subsampling_rate],
            'Y position': worm_data[:, 2][::self.subsampling_rate]
        })

    def classify_omegas(self, frames):
        print("Starting omega recognition.")
        for i, frame in enumerate(frames):
            solidity = get_solidity(self, frame)
            if solidity >= solidity_threshold:
                self.states[i] = state_dic["omega"]

    def classify_reversals(self, frames):
        print("Starting reversal recognition.")
        self.animal_length_um = 1000 * calculate_worm_size(self.path + frames[0], self)
        self.load_x_y_t()
        self.positional_data = calculate_reversals(self.positional_data, self.animal_length_um,
                                                   self.reversal_angle_threshold, self.scale_to_um)
        reversal_indices = np.where(self.positional_data['reversals'] == 1)[0]
        for rev_index in reversal_indices:
            if self.states[rev_index] == -1:
                self.states[rev_index] = state_dic["reversal"]

    def visualize_velocity(self):
        plt.plot(self.positional_data['velocity'])
        print(f'mean velocity: ', np.mean(self.positional_data['velocity']))
        plt.show()

    def classify_pauses(self):
        #for now, the pause is simply defined as the points s.t. v_i < 1/3 * mean(v)
        print("Starting pause recognition.")
        mean_vel =  np.mean(self.positional_data['velocity'])
        pause_indices = np.where(self.positional_data['velocity']<=1/3*mean_vel)[0]
        for p_index in pause_indices:
            if self.states[p_index] == -1:
                self.states[p_index] = state_dic["pause"]

    def classify_pirouettes(self):
        #a pirouette is the sequence of an omega and a reversal within 0.5s (2 frames)
        for i in range(len(self.states)-4):
            if self.states[i]==state_dic["omega"]:
                if state_dic["reversal"] in self.states[i+1:i+4]:
                    rev_idx = 0
                    for s in self.states[i+1:i+4]:
                        rev_idx+=1
                        if s==state_dic["reversal"]:
                            break
                    self.states[i:i+rev_idx+1] = state_dic["pirouette"]

    def classify(self):
        #load frames:
        frames = get_all_frame_paths(self.path, self.extension)[::self.subsampling_rate]
        self.states = np.zeros(len(frames)) - 1
        print("total n of possible states: ", len(self.states))
        #find the omegas
        self.classify_omegas(frames)
        #find reversals
        self.classify_reversals(frames)

        self.classify_pauses()

        self.classify_pirouettes()

        print("a total of:")
        unique, counts = np.unique(self.states, return_counts=True)
        state_dic_count = dict(zip(unique, counts))
        for key in state_dic_count:
            if key not in range(0, len(state_dic_count)):
                print("unlabeled state ", key, " appears ", state_dic_count[key], " times")
            else:
                print(f'state {key} ({list(state_dic.keys())[int(key)]}) appears {state_dic_count[key]} times')

