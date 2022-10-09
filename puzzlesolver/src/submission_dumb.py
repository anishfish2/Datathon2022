# DO NOT RENAME THIS FILE
# This file enables automated judging
# This file should stay named as `submission.py`

# Import Python Libraries
import numpy as np
from glob import glob
from PIL import Image
from itertools import permutations
from keras.models import load_model
from tensorflow.keras.utils import load_img, img_to_array

# Import helper functions from utils.py
import utils

class ImageBorders:
    def __init__(self, image_array):
        self.top_border = image_array[0]
        self.bottom_border = image_array[-1]
        self.left_border = image_array[:,0]
        self.right_border = image_array[:,-1]

class Predictor:
    """
    DO NOT RENAME THIS CLASS
    This class enables automated judging
    This class should stay named as `Predictor`
    """

    def __init__(self):
        """
        Initializes any variables to be used when making predictions
        """

    def make_prediction(self, img_path):
        """
        DO NOT RENAME THIS FUNCTION
        This function enables automated judging
        This function should stay named as `make_prediction(self, img_path)`

        INPUT:
            img_path: 
                A string representing the path to an RGB image with dimensions 128x128
                example: `example_images/1.png`
        
        OUTPUT:
            A 4-character string representing how to re-arrange the input imagie to solve the puzzle
            example: `3120`
        """

        # Load the image
        img = load_img(f'{img_path}', target_size=(128, 128))
        # Converts the image to a 3D numpy array (128x128x3)
        img_array = img_to_array(img)
        quadrants = utils.get_uniform_rectangular_split(img_array, 2, 2)
        quadrant_borderses = []
        for quadrant in quadrants:
            quadrant_borderses.append(ImageBorders(quadrant))

        perms = list(permutations([0, 1, 2, 3]))
       
        # Find tha score
        lowest_diff = 10000
        lowest_perm = None
        for permutation in perms:
            quadrant0 = permutation[0]
            quadrant1 = permutation[1]
            quadrant2 = permutation[2]
            quadrant3 = permutation[3]
            if np.mean(np.abs(quadrant_borderses[quadrant0].bottom_border - quadrant_borderses[quadrant2].top_border)) < 5:
                continue
            if np.mean(np.abs(quadrant_borderses[quadrant0].right_border - quadrant_borderses[quadrant1].left_border)) < 5:
                continue
            if np.mean(np.abs(quadrant_borderses[quadrant2].right_border - quadrant_borderses[quadrant3].left_border)) < 5:
                continue
            if np.mean(np.abs(quadrant_borderses[quadrant1].bottom_border - quadrant_borderses[quadrant3].top_border)) < 5:
                continue
            mean = (np.mean(np.abs(quadrant_borderses[quadrant0].bottom_border - quadrant_borderses[quadrant2].top_border)) + 
            np.mean(np.abs(quadrant_borderses[quadrant0].right_border - quadrant_borderses[quadrant1].left_border)) + 
            np.mean(np.abs(quadrant_borderses[quadrant2].right_border - quadrant_borderses[quadrant3].left_border)) + 
            np.mean(np.abs(quadrant_borderses[quadrant1].bottom_border - quadrant_borderses[quadrant3].top_border)))
            
            if mean < lowest_diff:
                lowest_diff = mean
                lowest_perm = permutation

        print("permutation: ", lowest_perm)





        # Return the combination that the example model thinks is the solution to this puzzle
        # Example return value: `3120`
        return "bingus"

# Example main function for testing/development
# Run this file using `python3 submission.py`
if __name__ == '__main__':

    for img_name in glob('example_images/*'):
        # Open an example image using the PIL library
        example_image = Image.open(img_name)

        # Use instance of the Predictor class to predict the correct order of the current example image
        predictor = Predictor()
        prediction = predictor.make_prediction(img_name)
        # Example images are all shuffled in the "3120" order
        print(prediction)

        # Visualize the image
        pieces = utils.get_uniform_rectangular_split(np.asarray(example_image), 2, 2)
        # Example images are all shuffled in the "3120" order
        final_image = Image.fromarray(np.vstack((np.hstack((pieces[3],pieces[1])),np.hstack((pieces[2],pieces[0])))))
        final_image.show()
