from texttable import Texttable
import numpy as np

def tab_printer(args):
    """
    Function to print the logs in a nice tabular format.
    input:
        param args: Parameters used for the model.
    """
    args = vars(args)
    keys = sorted(args.keys())
    t = Texttable() 
    t.add_rows([["Parameter", "Value"]] +  [[k.replace("_"," ").capitalize(), args[k]] for k in keys])
    print(t.draw())
    
def RMSE(ratings, ratings_hat, masks):
    """
    Input:
        - ratings:  numpy array matrix of actual user ratings.
        - ratings_hat: numpy array matrix of predicted user ratings.
        - test_mask:  0/1 test mask matrix. 
    Output:
        - This function returns the root mean square error (rmse) of the predictions. Note that the rmse should be computed using only those entries whose test_mask is equal to 1.
    """
    pass # replace by your code. 
