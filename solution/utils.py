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
     return (np.linalg.norm((ratings_hat - ratings)*masks, "fro") ** 2 / np.sum(masks)) ** 0.5
