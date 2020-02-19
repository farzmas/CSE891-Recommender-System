import argparse 
    
def parameter_parser():
    """
    all the parameter in the model.
    """
    parser = argparse.ArgumentParser(description = "Run reccomender system algorithms.")
    
    parser.add_argument("--input-file", nargs = "?",
                        default = "./data/jester.csv",
                        help = "path to input file.")
    
    parser.add_argument("--tau", type = float, default = None,
                        help = " parameter tau of the SVT matrix completion algorithm.")
    
    parser.add_argument("--delta", type = float, default =None,
                        help = " step size hyper-parameter for the SVT matrix completion algorithm.")
    
    parser.add_argument("--epsilon", type = float, default = 1e-2,
                        help = "the stopping criteria parameter for SVT matrix completion algorithm. Default is 1e-2")
    
    parser.add_argument("--lambda-1", type = float, default = 0.05,
                        help = "hyperparameters for sparsity regularization for U in MF algorithm. Default is 1e-2")
    
    parser.add_argument("--lambda-2", type = float, default = 0.05,
                        help = "hyperparameters for sparsity regularization for V in MF algorithm. Default is 1e-2")
    
    parser.add_argument("--max_iters", type = int, default = 3000,
                        help = "max_iters is maximum iteration count for the MC and MF algorithm. Defualt is 1000. ")
    
    parser.add_argument("--dim", type = int, default = 10,
                        help = "Number of columns for U and V in MF algorithm. Defualt is 10. ")    
    
    parser.add_argument("--test-size", type = float, default = 0.5,
                        help = "the proportion of the dataset to include in the test split. It should be between 0 and 1. Default is 0.5")
    
    parser.add_argument('--file', type=open, action=LoadFromFile)
    return parser.parse_args()

class LoadFromFile (argparse.Action):
    def __call__ (self, parser, namespace, values, option_string = None):
        with values as f:
            parser.parse_args(f.read().split(), namespace)