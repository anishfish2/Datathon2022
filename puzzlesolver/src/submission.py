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

def make_string_from_num_list(list):
    out = ""
    for item in list:
        out += str(item)
    return out 


def stitch_quadrants(img_array, quadrant_order):
    quadrants = utils.get_uniform_rectangular_split(img_array, 2, 2)
    stitched_top_row = np.concatenate((quadrants[quadrant_order[0]], quadrants[quadrant_order[1]]), axis=1)
    stitched_bottom_row = np.concatenate((quadrants[quadrant_order[2]], quadrants[quadrant_order[3]]), axis=1)
    stitched = np.concatenate((stitched_top_row, stitched_bottom_row), axis=0)
    return stitched


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
        self.model = load_model('src/submissionshuffle10000.h5')

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
            A 4-character string representing how to re-arrange the input image to solve the puzzle
            example: `3120`
        """

        # Load the image
        img = load_img(f'{img_path}', target_size=(128, 128))

        # Converts the image to a 3D numpy array (128x128x3)
        img_array = img_to_array(img)

        perms = list(permutations([0, 1, 2, 3]))
        most_confident_perm = None
        most_confident_perm_confidence = -1
        for perm in perms:
            perm_img_arr = stitch_quadrants(img_array, perm)
            # Convert from (128x128x3) to (Nonex128x128x3), for tensorflow
            img_tensor = np.expand_dims(perm_img_arr, axis=0)

            # Preform a prediction on this image using a pre-trained model (you should make your own model :))
            prediction = self.model.predict(img_tensor, verbose=False)
            confidence = prediction[0][0]
            if confidence > most_confident_perm_confidence:  # If the model thinks the image is unscrambled
                most_confident_perm_confidence = confidence
                most_confident_perm = perm
        
        #print(most_confident_perm)
        
        if most_confident_perm == None:
            return "0123"
        return make_string_from_num_list(most_confident_perm) 
        
        

        # The example model was trained to return the percent chance that the input image is scrambled using 
        # each one of the 24 possible permutations for a 2x2 puzzle
        #combs = [''.join(str(x) for x in comb) for comb in list(permutations(range(0, 4)))]

        # Return the combination that the example model thinks is the solution to this puzzle
        # Example return value: `3120`
        #return combs[np.argmax(prediction)]

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

        # # Visualize the image
        # pieces = utils.get_uniform_rectangular_split(np.asarray(example_image), 2, 2)
        # # Example images are all shuffled in the "3120" order
        # final_image = Image.fromarray(np.vstack((np.hstack((pieces[3],pieces[1])),np.hstack((pieces[2],pieces[0])))))
        # final_image.show()