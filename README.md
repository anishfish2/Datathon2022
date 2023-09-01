# Datathon 2022 - Puzzle Solver

![Datathon 2022 Logo](datathon_logo.png)

Welcome to the repository for the Datathon 2022 competition, where our primary focus was the Puzzle Solver challenge. In this competition, we aimed to unscramble images using various machine learning techniques.

## Puzzle Solver Challenge

Our task was to unscramble images, and we placed a particular emphasis on this aspect of the Datathon competition. Within the "Puzzle Solver" folder, you will find:

- `/scrambled_yep`: This folder contains scrambled images. Some of these images were provided by the competition organizers, while others were generated for additional training data.

- `/unscrambled_images`: Similar to the scrambled images, this folder holds unscrambled images. Some were provided, and others were generated for training purposes.

### Source Code

Within the `/src` folder, you will find a collection of files and scripts that contributed to our puzzle-solving efforts:

- `/example_images`: This directory contains a few example images to showcase our approach and results.

- `/models`: Here, you can find model checkpoints that store the weights and biases of our machine learning model.

- `datathon_jigsaw.py`: A script used for scrambling/unscrambling test images during training and evaluation.

- `eval.py`: A script to evaluate the performance of our model, utilizing binary classification to determine success and failure.

- `model.py`, `model2.py`, `model3.py`: These Python files contain the code for our machine learning models. `model.py` extracts image borders, while `model2.py` and `model3.py` perform the actual unscrambling using deep neural networks.

- `submission.h5`: Contains our final model, which achieved an impressive 93.5% accuracy on the final competition test set after six hours of training.

- `submission.py`: A file provided by the competition organizers for submitting the final model.

- `submission_dumb.py`, `submission_starter.py`, `submission_shuffle*.py`: Different iterations and checkpoints of our model throughout the competition.

### Understanding the Different `model.py` Files

Within the repository, you'll find three different versions of `model.py`. Here's a brief overview of their differences:

1. **`model.py`**: This script is responsible for extracting image borders from the scrambled pieces. It serves as the starting point for our puzzle-solving journey.

2. **`model2.py`**: In this version, we introduced a deep neural network that goes beyond border extraction. It was designed to match the borders of each image piece, finding the correct orientations for unscrambling. This model employs techniques such as Conv2D, batch normalization, dropout, and a final softmax layer with categorical cross-entropy and RMSprop for optimization.

3. **`model3.py`**: Similar to `model2.py`, this version focuses on matching borders but employs a shallower network with fewer layers. It offers an alternative approach to solving the puzzle with slightly different architecture choices.

## Accomplishing Puzzle Solving

Our approach to solving the puzzle involved the following key steps:

1. **Image Border Extraction**: We began by extracting borders from the scrambled images using the `model.py`. This step provided crucial information about the edges of each piece.

2. **Deep Neural Networks**: We employed deep neural networks, as implemented in `model2.py` and `model3.py`, to match the borders of each image piece. These models used techniques such as Conv2D, batch normalization, dropout, and a final softmax layer with categorical cross-entropy and RMSprop for optimization.

3. **Training Strategies**: To enhance training, we incorporated early stopping, learning rate plateau techniques, and utilized generators for training with a focus on efficiency.

4. **Model Evaluation**: We used `eval.py` to evaluate the performance of our models with a binary classification approach, distinguishing between successful and failed unscrambling attempts.

This led to an outstanding 93.5% accuracy on the final competition test set after six hours of training.

## Contributions

This project saw NitroGuy10's GitHub account hosting most of the commits. 

Thank you for exploring our Puzzle Solver repository. If you have any questions or feedback, please feel free to reach out.
