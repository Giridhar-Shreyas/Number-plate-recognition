from ultralytics import YOLO
import os
import utils as u

u.draw_pose(os.getcwd()+'\\data\\final_data\\test\\images\\Cars396_png.rf.609649ecb7c6b359d21b5b862aeb3d2f.jpg', [[0, 0.4471, 0.1191, 0.1888, 0.2382]])


"""

xywhn: tensor([[0.7968, 0.6885, 0.3312, 0.0867],
        [0.4471, 0.1191, 0.1888, 0.2382]], device='cuda:0')

"""