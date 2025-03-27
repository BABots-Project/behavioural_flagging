import json
from math import log

import cv2
import numpy as np
import pandas as pd
import scipy
from fontTools.subset import prune_hints
from matplotlib import pyplot as plt
from pandas import DataFrame
from scipy.ndimage import binary_fill_holes
from skimage.morphology import skeletonize, remove_small_objects
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from utils import get_all_frame_paths, get_solidity, calculate_worm_size, calculate_reversals, visualize_crawl_clusters, \
    visualize_single_crawl_cluster, get_first_frame_from_hdf5

#state_dic = {"omega": 0, "reversal": 1, "pirouette": 2, "pause":3, "line":4, "arc":5, "loop":6} #0 omega, 1 reversal, 2 pirouette, 3 pause, 4 line, 5 arc, 6 loop
state_dic = {"sharp_turn": 0, "reversal": 1, "pause":2, "line":3, "arc":4, "loop":5} #0 omega, 1 reversal, 2 pirouette, 3 pause, 4 line, 5 arc, 6 loop

solidity_threshold = 0.7 # minimum solidity for an sharp_turn


class Worm:
    def __init__(self, path:str, extension:str, reversal_angle_threshold:int, scale_to_um:float, subsampling_rate:float|int, pixel_to_mm, invert_color:bool, fps:float):
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
        self.reversal_angle_threshold = reversal_angle_threshold
        self.scale_to_um = scale_to_um
        self.subsampling_rate = int(subsampling_rate)
        self.pixel_to_mm_ratio = pixel_to_mm
        self.invert_color = invert_color
        self.positional_data = []
        self.sharp_turn_indices = []
        self.reversal_indices = []
        self.pirouette_indices = []
        self.pause_indices = []
        self.all_frames = []#get_all_frame_paths(self.path, self.extension)[::self.subsampling_rate]
        self.head_idx = -1
        self.midpoint_idx = -1
        self.load_x_y_t_zenodo()
        self.states = np.zeros(len(self.positional_data['Timestamp'])) - 1
        print(f'expecting a total of {len(self.states)} frames')
        print(f'Recording time = {(self.positional_data['Timestamp'].iloc[-1] - self.positional_data['Timestamp'].iloc[0])/60} min')
        delta_t = self.positional_data['Timestamp'].iloc[1] - self.positional_data['Timestamp'].iloc[0]
        print(f'Delta t = {delta_t} s')
        print(f'fps = {1/(delta_t)}')
        self.sharp_turn_threshold = 2/3 * np.pi
        self.angles = np.zeros(len(self.positional_data['Timestamp'])) - 1
        #perform vision algo iff head==midpoint (no segmentation)
        if self.head_idx == self.midpoint_idx:
            first_frame = get_first_frame_from_hdf5(self.path.split(".")[0] + ".hdf5")
            length_from_first_frame = calculate_worm_size(first_frame, self) / 1000
            print(f'estimated len={length_from_first_frame}')
            self.animal_length_um = min(length_from_first_frame , 1100)
            print(f'animal length = {self.animal_length_um} um')
        else:
            self.animal_length_um = 1000
        # -1.0
        self.fps = fps
        print("SUBSAMPLING RATE: ", self.subsampling_rate)


    def load_x_y_t_zenodo(self):
        with open(self.path) as f:
            worm_data = json.load(f)
        #print("the first 32-non filtered timestamps are: ", worm_data["data"][0]["t"][:32])
        #print("from which, the first 4 filtered timestamps are: ", worm_data["data"][0]["t"][::self.subsampling_rate][:4])
        #print("worm data 0 x: ", worm_data["data"][0]["x"])
        print("the worm is segmented into ", len(worm_data["data"][0]["x"][0]), " segments")
        print("head is on the : ", worm_data["data"][0]["head"])
        head_or = worm_data["data"][0]["head"]
        if head_or =="L" or head_or=="l":
            head_idx = 0
        else:
            head_idx = len(worm_data["data"][0]["x"][0]) - 1
        print("choosing head index ", head_idx)
        midpoint_idx = len(worm_data["data"][0]["x"][0]) // 2
        self.head_idx = head_idx
        self.midpoint_idx = midpoint_idx
        self.positional_data = pd.DataFrame({
            'Timestamp': worm_data["data"][0]["t"][::self.subsampling_rate], #s
            'X position': [0.001 * x[midpoint_idx] if x[midpoint_idx] is not None else None for x in worm_data["data"][0]["x"][::self.subsampling_rate] ],  # mm
            'Y position': [0.001 * y[midpoint_idx] if y[midpoint_idx] is not None else None for y in worm_data["data"][0]["y"][::self.subsampling_rate]],   # mm
            'X head': [0.001 * x[head_idx] if x[head_idx] is not None else None for x in worm_data["data"][0]["x"][::self.subsampling_rate] ],  # mm
            'Y head': [0.001 * y[head_idx] if y[head_idx] is not None else None for y in worm_data["data"][0]["y"][::self.subsampling_rate]]
        })
        #print("Timestamp: ", self.positional_data["Timestamp"])
        self.positional_data.dropna(inplace=True)
        #print("Timestamp 2: ", self.positional_data["Timestamp"])


    def load_x_y_t(self):
        worm_data = np.loadtxt(self.path + "data.txt", delimiter="\t")

        self.positional_data = pd.DataFrame({
            'Timestamp': worm_data[:, 0][::self.subsampling_rate],
            'X position': worm_data[:, 1][::self.subsampling_rate],
            'Y position': worm_data[:, 2][::self.subsampling_rate]
        })

    def load_x_y_t_euphrasie(self):
        with open(self.path, "r") as f:
            lines = [line.rstrip() for line in f]
        worm_data = np.loadtxt(lines, delimiter=" ", skiprows=28)
        self.positional_data = pd.DataFrame({
            'Timestamp': worm_data[:, 1][::self.subsampling_rate] * 1e-3,
            'X position': worm_data[:, 2][::self.subsampling_rate],
            'Y position': worm_data[:, 3][::self.subsampling_rate]
        })


    def classify_omegas(self):
        print("Starting omega recognition.")
        for i, frame in enumerate(self.all_frames):
            solidity, thresh = get_solidity(self, frame)
            if solidity >= solidity_threshold:
                self.states[i] = state_dic["omega"]
                self.omega_indices.append(i)
            else:
                skeleton = skeletonize(thresh)
                skeleton = remove_small_objects(skeleton, min_size=max(10, int(self.animal_length_um/2)), connectivity=2)
                filled_skeleton = binary_fill_holes(skeleton)
                if (skeleton!=filled_skeleton).any():
                    self.states[i] = state_dic["omega"]
                    self.omega_indices.append(i)

    def classify_sharp_turns(self):
        angle_changes =np.diff(self.angles)
        angle_changes = [angle_changes[i] - 2 * np.pi if angle_changes[i] > np.pi else angle_changes[i] for i in
                          range(len(angle_changes))]
        angle_changes = [angle_changes[i] + 2 * np.pi if angle_changes[i] < -np.pi else angle_changes[i] for i in
                          range(len(angle_changes))]
        print("average angle change: ", np.mean(angle_changes))
        print("max angle change: ", np.max(angle_changes))
        print("min angle change: ", np.min(angle_changes))

        for idx, change in enumerate(angle_changes):
            frame_index = idx + 2  # shift by 2 to align with the transitions from the heading calculation
            if abs(change) >= self.sharp_turn_threshold:
                if self.states[frame_index] == -1:
                    self.states[frame_index] = state_dic["sharp_turn"]
                    self.sharp_turn_indices.append(frame_index)



    def classify_reversals(self):
        print("Starting reversal recognition.")
        if self.animal_length_um<0:
            self.animal_length_um = 1000 * calculate_worm_size(self.path + self.all_frames[0], self)

        self.positional_data = calculate_reversals(self.positional_data, self.animal_length_um,
                                                   self.reversal_angle_threshold, self.scale_to_um)
        dx = np.diff(self.positional_data['X_um'])
        dy = np.diff(self.positional_data['Y_um'])
        angles = np.arctan2(dy, dx)
        self.angles = angles
        reversal_indices = np.where(self.positional_data['reversals'] == 1)[0]
        self.reversal_indices = list(reversal_indices)
        for rev_index in reversal_indices:
            if self.states[rev_index] == -1:
                self.states[rev_index] = state_dic["reversal"]

    def classify_reversals_segmented(self):
        """
        Identify reversal frames by checking if the velocity vector is opposite
        to the head-to-midpoint vector.
        """
        self.positional_data['X_um'] = self.positional_data['X position'] * self.scale_to_um
        self.positional_data['Y_um'] = self.positional_data['Y position'] * self.scale_to_um
        self.positional_data['time_diff'] = self.positional_data['Timestamp'].diff().fillna(1)  # Assume a default time difference for the first frame

        # Calculate velocities (micrometers per second)
        self.positional_data['velocity'] = np.sqrt(
            (self.positional_data['X_um'].diff() ** 2) + (self.positional_data['Y_um'].diff() ** 2)
        ) / self.positional_data['time_diff']
        self.positional_data = self.positional_data.fillna(0)
        dx = np.diff(self.positional_data['X_um'])
        dy = np.diff(self.positional_data['Y_um'])
        angles = np.arctan2(dy, dx)
        print("calculated angles: ", angles)
        self.angles = angles
        # Compute head-to-midpoint vectors
        head_x = self.positional_data['X head'].values
        head_y = self.positional_data['Y head'].values
        mid_x = self.positional_data['X position'].values
        mid_y = self.positional_data['Y position'].values

        head_to_mid_x = mid_x - head_x
        head_to_mid_y = mid_y - head_y

        # Compute velocity vectors
        vel_x = np.diff(mid_x, prepend=mid_x[0]) / np.diff(self.positional_data['Timestamp'], prepend=1)
        vel_y = np.diff(mid_y, prepend=mid_y[0]) / np.diff(self.positional_data['Timestamp'], prepend=1)

        # Normalize vectors
        head_to_mid_norm = np.sqrt(head_to_mid_x ** 2 + head_to_mid_y ** 2)
        vel_norm = np.sqrt(vel_x ** 2 + vel_y ** 2)

        # Avoid division by zero
        head_to_mid_norm[head_to_mid_norm == 0] = 1
        vel_norm[vel_norm == 0] = 1

        head_to_mid_x /= head_to_mid_norm
        head_to_mid_y /= head_to_mid_norm
        vel_x /= vel_norm
        vel_y /= vel_norm

        # Compute dot product to check for reversals
        dot_product = head_to_mid_x * vel_x + head_to_mid_y * vel_y

        plt.hist(np.degrees(np.arccos(dot_product)), bins=30)
        plt.xlabel("Angle (degrees)")
        plt.ylabel("Frequency")
        plt.title("Distribution of Head-to-Midpoint vs Velocity Angles")
        plt.show()
        plt.close()

        print("Sample dot product values:", dot_product[:20])
        print("Reversal threshold:", np.cos(np.radians(self.reversal_angle_threshold)))

        # Reversals occur when the dot product is negative (vectors pointing in opposite directions)
        self.reversal_indices = np.where(dot_product < np.cos(np.radians(self.reversal_angle_threshold)))[0]
        print("apparently there are this many reversals: ", len(self.reversal_indices))
        self.states[self.reversal_indices] = state_dic["reversal"]


    def visualize_velocity(self):
        #plt.plot(self.positional_data['velocity'])
        print(f'mean velocity: ', np.mean(self.positional_data['velocity']))
        #plt.show()

        plt.hist(self.positional_data["velocity"], bins=50)
        plt.show()

    def classify_pauses(self):
        #for now, the pause is simply defined as the points s.t. v_i < 1/2 * mode(v)
        print("Starting pause recognition.")
        #mean_vel =  scipy.stats.mode(self.positional_data['velocity']).mode
        #mean_vel =0.5* self.positional_data['velocity'].quantile(0.25)
        #mean_vel = np.mean(self.positional_data['velocity']) - 2.0 * np.std(self.positional_data['velocity'])
        #mean_vel = np.percentile(self.positional_data['velocity'], 0.2)
        #fixed value: pause is NOT moving, thus, the displacement should be <=50um * fps, where fps is the ACTUAL fps/subsampling_rate
        #mean_vel = 50 * self.fps / self.subsampling_rate
        #print("speed threshold: ", mean_vel)
        #pause_indices = np.where(self.positional_data['velocity']<=
         #                        mean_vel)[0]
        #fixed value: pause is NOT moving, thus, the displacement should be <=50um/dt

        pause_indices = np.where(self.positional_data['velocity'] <=
                                 50 / self.positional_data['time_diff'])[0]
        self.pause_indices = pause_indices
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
                    for j in range(i, i+rev_idx+1):
                        if j in self.reversal_indices:
                            self.reversal_indices.remove(j)
                        elif j in self.omega_indices:
                            self.omega_indices.remove(j)
                        self.pirouette_indices.append(j)
            elif self.states[i]==state_dic["reversal"]:
                if state_dic["omega"] in self.states[i+1:i+4]:
                    omega_idx = 0
                    for s in self.states[i+1:i+4]:
                        omega_idx+=1
                        if s==state_dic["omega"]:
                            break
                    self.states[i:i+omega_idx]=state_dic["pirouette"]



    def get_crawl_indices(self):
        crawl_start_end_indices = []
        i = 0
        while i < len(self.states):
            if self.states[i] == -1:  # not in a sequence AND starting sequence
                start = i
                if i+1<len(self.states):
                    i+=1
                    while self.states[i] == -1:
                        i += 1
                        if i>=len(self.states):
                            break
                    end = i
                else:
                    end = i
                crawl_start_end_indices.append([start, end])
                i+=1
            else:
                i += 1
        return crawl_start_end_indices

    def get_crawl_curvature(self, start_index, end_index):
        dx_dt = np.gradient(self.positional_data['X_um'][start_index:end_index])
        dy_dt = np.gradient(self.positional_data['Y_um'][start_index:end_index])
        d2x_dt2 = np.gradient(dx_dt)
        d2y_dt2 = np.gradient(dy_dt)
        if (dx_dt * dx_dt + dy_dt * dy_dt).all()<=1e-16:
            print("cooked from ", start_index, end_index)
            curvature = [1]
        else:
            curvature = np.abs(d2x_dt2 * dy_dt - dx_dt * d2y_dt2) / (dx_dt * dx_dt + dy_dt * dy_dt) ** 1.5
        avg_curv = np.mean(curvature)
        return avg_curv


    def get_angular_concordance(self, start_index, end_index):
        dx = np.diff(self.positional_data['X_um'][start_index:end_index])
        dy = np.diff(self.positional_data['Y_um'][start_index:end_index])
        angles = np.arctan2(dy, dx)
        cos_angles = np.cos(angles)
        sin_angles = np.sin(angles)
        mean_x = np.mean(cos_angles)
        mean_y = np.mean(sin_angles)
        r_a = np.sqrt(mean_x ** 2 + mean_y ** 2)
        return r_a

    def visualize_crawl_data(self, curvatures, concordances):
        #plt.scatter(concordances, [log(curv) if curv>0 else 0 for curv in curvatures])
        plt.scatter(concordances, np.log(curvatures))
        plt.xlabel("Angular concordance")
        plt.ylabel("Curvature (ln(k))")
        plt.show()


    def cluster_crawls(self, curvatures, concordances):
        #print("curvatures: ", curvatures)
        #crawl_data = [[conc, log(curv)] if curv>0 else [conc, 0] for curv,conc in zip(curvatures, concordances) ]
        crawl_data = [[conc, curv] for curv, conc in zip(curvatures, concordances)]
        print("there are this many 0 curvature points:")
        print(len(curvatures) - np.count_nonzero(curvatures))
        inertias = []
        labels = []
        centers = []
        min_sil_k = -1
        min_sil_score = np.inf
        for i in range(2, 11):
            kmeans = KMeans(n_clusters=i)
            kmeans.fit(crawl_data)
            inertias.append(kmeans.inertia_)
            labels.append(kmeans.labels_)
            centers.append(kmeans.cluster_centers_)
            sil = silhouette_score(crawl_data, kmeans.labels_)
            print("for k=", i, " silhouette score is ", sil)
            if sil<min_sil_score:
                min_sil_score = sil
                min_sil_k = i
        print("min silhouette for k=", min_sil_k, " with score ", min_sil_score)
        visualize_crawl_clusters(concordances, curvatures, labels, centers[:min_sil_k+1], min_sil_k)

    def get_clustered_crawls(self, curvatures, concordances, k=3):
        crawl_data = [[conc, curv] for curv, conc in zip(curvatures, concordances)]
        kmeans = KMeans(n_clusters=k)
        if len(crawl_data)<3:
            print("Not enough crawl data present. len of conc, len of curv: ", len(concordances), len(curvatures))
            return [[], []]
        kmeans.fit(crawl_data)
        labs = kmeans.labels_
        centers = kmeans.cluster_centers_
        sorted_centers = sorted(centers, key=lambda x: x[0], reverse=True) # ordered by increasing concordance: 0->lines, 1->arcs, 2->loops
        print("Found labs: ", labs)
        print("Corresponding centers: ", centers)
        visualize_single_crawl_cluster(concordances, curvatures, labs, centers)
        point_ids_by_label = []
        label_to_state_id = [] #same len as point_ids_by_label, converts the positions to 3,4,5 (lines,arcs,loops)
        for i in range(k):
            p_indices = np.where(labs == i)[0]
            center_concordance = centers[i][0]
            concordance_index = np.where(sorted_centers == center_concordance)[0]
            print(f'there are {len(p_indices)} for label {i}')
            point_ids_by_label.append(p_indices)
            label_to_state_id.append(concordance_index+3)
        return point_ids_by_label, label_to_state_id

    def classify_crawls(self):
        print("Starting crawl recognition.")
        crawl_start_end_indices = self.get_crawl_indices()
        curvatures = []
        angular_concordances = []
        x_crawls = []
        y_crawls = []
        valid_crawl_start_end_indices = []
        for start, end in crawl_start_end_indices:
            if end-start>1: #ignore size 1 crawls (happens at the beginning)
                curvature = self.get_crawl_curvature(start, end)
                if 1e-10<curvature<1: #ignore above 0 ln(k) for now
                    angular_concordance = self.get_angular_concordance(start, end)
                    curvatures.append(curvature)
                    angular_concordances.append(angular_concordance)
                    x_crawls.append(self.positional_data["X_um"][start:end])
                    y_crawls.append(self.positional_data["Y_um"][start:end])
                    valid_crawl_start_end_indices.append([start, end])
                else:
                    self.states[start:end] = state_dic["arc"]
            else:
                self.states[start:end] = state_dic["line"]
        self.visualize_crawl_data(curvatures, angular_concordances)
        #self.cluster_crawls(curvatures, angular_concordances)
        if len(curvatures)==0:
            print('No curvatures found')
            for i in state_dic.values():
                if i in self.states:
                    print(f'state {list(state_dic.keys())[int(i)]} appears {len(np.where(self.states == int(i))[0])} times')
            print(f'unlabeled state appears {len(np.where(self.states == int(-1))[0])} times')
            self.visualize_velocity()
            return

        p_by_ids, l_to_s = self.get_clustered_crawls(curvatures, angular_concordances, 3)
        for point_by_id, label_to_state in zip(p_by_ids, l_to_s):
            for point in point_by_id:
                start, end = valid_crawl_start_end_indices[point]
                for i in range(start, end):
                    if self.states[i]==-1:
                        self.states[i] = int(label_to_state)


    def log_states(self, filename):
        np.save(filename, self.states)

    def log_positions(self, filename):
        xs = self.positional_data["X_um"]
        ys = self.positional_data["Y_um"]
        np.save(filename, [xs, ys])

    def log_velocities(self, filename):
        np.save(filename, self.positional_data["velocity"])


    def classify(self):
        if self.head_idx != self.midpoint_idx:
            self.classify_reversals_segmented()
        else:
            self.classify_reversals()
        #self.classify_omegas()
        #self.classify_pirouettes()
        self.classify_pauses()
        self.classify_sharp_turns()

        self.classify_crawls()
        print("a total of:")
        unique, counts = np.unique(self.states, return_counts=True)
        state_dic_count = dict(zip(unique, counts))
        for key in state_dic_count:
            if key not in range(0, len(state_dic)):
                print("unlabeled state ", key, " appears ", state_dic_count[key], " times")
            else:
                print(f'state {key} ({list(state_dic.keys())[int(key)]}) appears {state_dic_count[key]} times')

