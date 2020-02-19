import numpy as np
from sklearn.model_selection import train_test_split
def load_data(input_file, test_size, random_seed ):
    """
    Function to load the data and create train an test sets.
    Input:
        - input_file: path to input file.
        - test_size: float between 0.0 and 1.0 and represent the proportion of the dataset to include in the test split.
        - random_seed:  is the integer seed used by the random number generator. 
    Output:
        - Ratings: n(number of users) by m(number of books) numpy array of user ratings.
        - train_masks:  n by m numpy array matrix with element 0 and 1. Element(u,b) is 1 if the user u rating for item b belongs to train set) 0 (if it's not).
        - test_masks: n by m numpy array matrix with element 0 and 1. Element(u,b) is 1 if the user u rating for item b belongs to test set) 0 (if it's not).
    """
    ratings = np.genfromtxt(input_file, delimiter=',')
    #ratings = ratings[:200,:]
    train_masks = np.zeros_like(ratings)
    test_masks = np.zeros_like(ratings)
    pairs = list()
    for i in range(ratings.shape[0]):
        for j in range(ratings.shape[1]):
            pairs.append((i,j))
    train, test = train_test_split(pairs ,test_size = test_size, random_state= random_seed)
    for user,item in train:
        train_masks[user, item]= 1
    for user,book in test:
        test_masks[user, item]= 1   
    return ratings, train_masks, test_masks
    
    