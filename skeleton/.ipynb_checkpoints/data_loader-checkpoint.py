import numpy as np
def load_data(input_file, test_size, random_seed ):
    """
    Function to load the data and create train an test sets.
    Input:
        - input_file: path to input file.
        - test_size: a floating point number between 0.0 and 1.0, which represents the proportion of 
        the input data to be used as the test set.
        - random_seed:  an integer seed to be used by the random number generator. 
    Output:
        - Ratings: A numpy array of user ratings of size n(number of users) by m(number of items).
        Each element(u,b) is the rating of user u for the item b. 
        - train_mask:  A binary 0/1 numpy array matrix of size n by m. 
        Element(u,b) is set to 1 if the user-item pair (u,b) belongs to the training set; otherwise, it is set to 0.
        - test_mask: A binary 0/1 numpy array matrix of size n by m. 
        Element(u,b) is set to 1 if the user-item pair (u,b) belongs to the test set; otherwise, it is set to 0.    
    """
    pass
    
    