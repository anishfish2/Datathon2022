import numpy as np
from glob import glob
from PIL import Image
from itertools import permutations
from keras.models import load_model
from tensorflow.keras.utils import load_img, img_to_array
import pandas as pd
# Import helper functions from utils.py
import utils
import os

training_data_folder = '../train'

class ImageBorders:
    def __init__(self, image_array):
        self.top_border = image_array[0]
        self.bottom_border = image_array[-1]
        self.left_border = image_array[:,0]
        self.right_border = image_array[:,-1]

if __name__ == '__main__':
    total_data = pd.DataFrame(columns= ['topborder0', 'rightborder0', "leftborder0", "bottomborder0", 
                                "topborder1", "rightborder1", "leftborder1", "bottomborder1",
                                "topborder2", "rightborder2", "leftborder2", "bottomborder2", 
                                "topborder3", "rightborder3", "leftborder3", "bottomborder3", "correct_sequence" ]
    )
    
    for folder_name in os.listdir(training_data_folder):
        for img_name in os.listdir(os.path.join(training_data_folder, folder_name)):
            img_path = os.path.join(training_data_folder, folder_name, img_name)
            correct_sequence = folder_name

            img = load_img(f'{img_path}', target_size=(128, 128))
            # Converts the image to a 3D numpy array (128x128x3)
            img_array = img_to_array(img)
            quadrants = utils.get_uniform_rectangular_split(img_array, 2, 2)
            quadrant_borderses = []
            for quadrant in quadrants:
                quadrant_borderses.append(ImageBorders(quadrant))
            
            # get quadrant 4's top border
            quadrant_borderses[3].top_border
            total_data.loc[len(total_data.index)] = [
                quadrant_borderses[0].top_border, quadrant_borderses[0].right_border, quadrant_borderses[0].left_border, quadrant_borderses[0].bottom_border,
                quadrant_borderses[1].top_border, quadrant_borderses[1].right_border, quadrant_borderses[1].left_border, quadrant_borderses[1].bottom_border,
                quadrant_borderses[2].top_border, quadrant_borderses[2].right_border, quadrant_borderses[2].left_border, quadrant_borderses[2].bottom_border,
                quadrant_borderses[3].top_border, quadrant_borderses[3].right_border, quadrant_borderses[3].left_border, quadrant_borderses[3].bottom_border, correct_sequence
                ]
            # total_data.append({'topborder0' : quadrant_borderses[0].top_border, 'rightborder0' : quadrant_borderses[0].right_border, "leftborder0" : quadrant_borderses[0].left_border, "bottomborder0" : quadrant_borderses[0].bottom_border,
            #                                 "topborder1" : quadrant_borderses[1].top_border, "rightborder1" : quadrant_borderses[1].right_border, "leftborder1" :quadrant_borderses[1].left_border, "bottomborder1" :quadrant_borderses[1].bottom_border,
            #                                 "topborder2" : quadrant_borderses[2].top_border, "rightborder2" :quadrant_borderses[2].right_border, "leftborder2" :quadrant_borderses[2].left_border, "bottomborder2" :quadrant_borderses[2].bottom_border,
            #                                 "topborder3" :quadrant_borderses[3].top_border, "rightborder3" :quadrant_borderses[3].right_border, "leftborder3" :quadrant_borderses[3].left_border, "bottomborder3" :quadrant_borderses[3].bottom_border, "correct_sequence" : correct_sequence}, ignore_index = True)
            print(total_data)
        