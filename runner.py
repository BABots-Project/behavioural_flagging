import os.path
import sys
import time
from math import ceil

import numpy as np

from utils import get_solidity, calculate_worm_size
from worm import Worm
OVERWRITE = False
DESIRED_FPS = 4 # as per Salvador
frame_folders = [
    "Foraging/mb01_056_N2tmd24/",
    "Foraging/mb01_055_N2tmd20/",
    "Some data _ Euphrasie/2024-02-07-13-58-26-coords.txt",
    "Some data _ Euphrasie/2025-01-31-15-57-21-007648-coords.txt",
    "Some data _ Euphrasie/2025-01-31-16-03-00-891561-coords.txt",
    "zenodo_data/N2 on food R_2010_01_21__11_01_19___6___1.wcon", #fps=31.9489, video micrometers per pixel : 4.64175
    "zenodo_data/N2 on food L_2011_02_17__11_39_45___7___3.wcon", #fps=30.03, video micrometers per pixel : 4.2072
    "zenodo_data/N2_on food_R_2014_02_05__15_55_40___7___.wcon", #fps=30.03, video micrometers per pixel : 4.12939
    "zenodo_data/N2 on food R_2012_08_30__12_35_33__4.wcon", #26.3158, video micrometers per pixel : 4.02703
    "zenodo_data/N2 on food L_2011_10_21__11_17_03___2___4.wcon" # 30.03, 4.42061
]

zenodo_frame_folders = [
    "zenodo_data/N2 on food R_2010_01_21__11_01_19___6___1.wcon", #fps=31.9489, video micrometers per pixel : 4.64175
    "zenodo_data/N2 on food L_2011_02_17__11_39_45___7___3.wcon", #fps=30.03, video micrometers per pixel : 4.2072
    "zenodo_data/N2_on food_R_2014_02_05__15_55_40___7___.wcon", #fps=30.03, video micrometers per pixel : 4.12939
    "zenodo_data/N2 on food R_2012_08_30__12_35_33__4.wcon", #26.3158, video micrometers per pixel : 4.02703
    "zenodo_data/N2 on food L_2011_10_21__11_17_03___2___4.wcon", # 30.03, 4.42061
    "zenodo_data/N2 on food R_2010_01_12__10_39_50___6___1.wcon",
    "zenodo_data/N2 on food R_2014_02_04__11_45_18___7___1.wcon",
    "zenodo_data/N2 on food R_2009_07_15__10_36_13__1.wcon",
    "zenodo_data/N2 on food R_2016_02_03__11_52_47___3___3.wcon",
    "zenodo_data/N2 on food R_2010_02_24__09_23_04___6___1.wcon",
    "zenodo_data/N2 on food L_2011_02_04__15_01___3___9.wcon",
    "zenodo_data/N2 on food R_2012_10_04__11_44_17___8___8.wcon",
    "zenodo_data/N2 on food R_2010_03_26__10_14_56___7___1.wcon",
    "zenodo_data/N2 on food R_2015_02_18__12_50_05___4___6.wcon",
    "zenodo_data/N2 on food L_2012_08_29__12_13_17___5___1.wcon",
    "zenodo_data/N2 on food R_2011_09_22__13_36_02___7___8.wcon",
    "zenodo_data/N2 on food L_2016_02_03__11_30_03___5___2.wcon",
    "zenodo_data/N2 on food R_2009_12_11__12_01___3___1.wcon",
    "zenodo_data/N2 on food L_2011_08_19__13_21___4___9.wcon",
    "zenodo_data/N2 on food R_2009_12_11__12_02_51___4___1.wcon",
    "zenodo_data/N2 on food on food R_2011_10_21__11_39___3___5.wcon",
    "zenodo_data/N2 on food L_2011_03_29__17_02_16__17.wcon",
    "zenodo_data/N2 on food R_2010_07_16__10_27_41__1.wcon",
    "zenodo_data/N2 on food R_2012_10_23__10_23_39___6___4.wcon",
    "zenodo_data/N2 on food R_2012_08_31__12_41_18___7___7.wcon",
    "zenodo_data/N2 on food R_2011_11_11__12_34___4___2.wcon",
    "zenodo_data/N2 on food L_2011_03_30__11_38_43___6___1.wcon",
    "zenodo_data/N2 on food R_2010_09_24__11_18_27___7___1.wcon",
    "zenodo_data/N2 on food R_2013_03_21__10_32_31___7___1.wcon",
    "zenodo_data/N2 on food L_2009_09_04__11_12_10___4___5.wcon",
    "zenodo_data/N2 on food R_2010_08_03__10_15_12___2___1.wcon",
    "zenodo_data/N2 on food L_2010_06_15__10_25_51___4___1.wcon",
    "zenodo_data/N2 on food R_2012_04_17__15_38_48___4___2.wcon",
    "zenodo_data/N2 on food R_2012_10_31__11_35_09___1___6.wcon",
    "zenodo_data/N2 on food L_2011_09_22__11_29_07___1___2.wcon",
    "zenodo_data/N2 on food R_2011_05_19__11_59___3___4.wcon",
    "zenodo_data/N2 on food R_2010_10_15__15_34___3___10.wcon"
]

zenodo_links = ["https://zenodo.org/records/1029802", #page 3 start
                "https://zenodo.org/records/1029802",
                "https://zenodo.org/records/1029802",
                "https://zenodo.org/records/1029802",
                "https://zenodo.org/records/1021768",
                "https://zenodo.org/records/1003993",
                "https://zenodo.org/records/1014992",
                "https://zenodo.org/records/1032645",
                "https://zenodo.org/records/1033903",
                "https://zenodo.org/records/1005760",
                "https://zenodo.org/records/1007403", #page 3 end
                "https://zenodo.org/records/1028649", #page 1 start
                "https://zenodo.org/records/1030739",
                "https://zenodo.org/records/1006588",
                "https://zenodo.org/records/1009360",
                "https://zenodo.org/records/1017304",
                "https://zenodo.org/records/1007167",
                "https://zenodo.org/records/1027697",
                "https://zenodo.org/records/1009398",
                "https://zenodo.org/records/1008014" #page 1 end
                "https://zenodo.org/records/1031222", #"off" food (really on food) pages 3-4 start
                "https://zenodo.org/records/1010398",
                "https://zenodo.org/records/1029162",
                "https://zenodo.org/records/1019291",
                "https://zenodo.org/records/1022874",
                "https://zenodo.org/records/1008012",
                "https://zenodo.org/records/1020603",
                "https://zenodo.org/records/1013745",
                "https://zenodo.org/records/1016886",
                "https://zenodo.org/records/1032995",
                "https://zenodo.org/records/1017316",
                "https://zenodo.org/records/1029641",
                "https://zenodo.org/records/1008536",
                "https://zenodo.org/records/1022976",
                "https://zenodo.org/records/1022113",
                "https://zenodo.org/records/1008738",
                "https://zenodo.org/records/1015930" #end

]
zenodo_fps = [
    31.9489, 30.03, 30.03, 26.3158, 30.03, 32.1543, 30.03, 25.9067, 20.0, 25.4453,
    30.03, 24.0385,25.2525,30.03,30.03, 30.03, 30.03, 26.5252, 30.03, 26.3158,
20.0, 30.03, 26.0417, 30.03, 30.03, 30.03, 30.03, 30.03, 30.03, 27.248, 30.03,
25.7069, 30.03, 30.03, 30.03, 30.03, 30.03
]
zenodo_px_to_um = [
4.64175, 4.2072, 4.12939, 4.02703, 4.42061, 4.64514, 4.12939, 4.83785, 4.61106, 4.33523,
4.72886, 4.06927,4.31067,4.65615,  4.30998, 4.18125, 6.08706, 4.75979, 4.76371, 4.69255,
4.6309, 4.45319, 4.56732, 4.17406, 4.15398, 4.73669, 4.52262, 4.14723, 4.12939, 5.2418,
4.49333, 4.7068, 4.48097, 4.53154, 4.44972, 4.72886, 4.53292
]

#print( len(zenodo_frame_folders), len(zenodo_links), len(zenodo_fps), len(zenodo_px_to_um))
assert len(zenodo_fps) == len(zenodo_px_to_um) == len(zenodo_frame_folders), print("Something is wrong -- check length of zenodo arrays")

print("total number of files: ", len(zenodo_fps))
idx = 2
frame_folder = zenodo_frame_folders[idx]

def main(i=-1):
    global frame_folder
    frame_folder = zenodo_frame_folders[i]
    # 1/405
    # 1.67158244675651
    pixel_to_mm_ratio = zenodo_px_to_um[i] * 1000  # 1/405
    stepper_to_mm_ratio = 1  # 1/788 #1
    fps = zenodo_fps[i]
    subsampling_rate = ceil(fps / DESIRED_FPS)
    worm = Worm(frame_folder, '.jpg', 150, stepper_to_mm_ratio*1000, subsampling_rate, pixel_to_mm_ratio, True, fps)
    start_time = time.time()
    #sys.exit()
    worm.classify()
    worm.visualize_velocity()
    if not os.path.isdir(f"states/{frame_folder.split("/")[0]}"):
        os.mkdir(f"states/{frame_folder.split("/")[0]}")
    if not os.path.isdir(f"states/{frame_folder}"):
        os.mkdir(f"states/{frame_folder}")
    worm.log_states(f"states/{frame_folder}/states")
    worm.log_positions(f"states/{frame_folder}/positions")
    worm.log_velocities(f"states/{frame_folder}/velocities")
    #print(get_solidity(worm, worm.all_frames[2131]))
    end_time = time.time()

    print("that took: ", end_time-start_time, " seconds.")


if __name__=="__main__":
    #main(2)
    #sys.exit()
    for i in range(len(zenodo_fps)):
        print("classifying worm ", i)
        main(i)