import time

from worm import Worm

DESIRED_FPS = 4 # as per Salvador

frame_folder = "Foraging/mb01_055_N2tmd20/"
pixel_to_mm_ratio = 1/405
stepper_to_mm_ratio = 1/788
fps=32

worm = Worm(frame_folder, '.jpg', 120, stepper_to_mm_ratio*1000, fps//DESIRED_FPS, pixel_to_mm_ratio, True)
start_time = time.time()
worm.classify()
end_time = time.time()

print("that took: ", end_time-start_time, " seconds.")
