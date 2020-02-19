import numpy as np

"""
Base Recommender class. This is the base class for Mean Imputation(MI), Matrix completion(MC), and Matrix Factorization (MF) classes. You do not need to modify this class.
"""
class Recommender(object):
    def __init__(self,ratings):
        self.ratings = ratings # actual rating matrix 
    
    
    def SVD(self, X): 
        """
         SVD decomposition function         
        """
        return np.linalg.svd(X, full_matrices=False)


        
"""
Mean Imputation approach. You need to implement fit and predict functions. 
"""
class MI(Recommender): 
    def __init__(self,ratings):
        Recommender.__init__(self, ratings) 
        self.rating_avg = None # The Mean Imputation model parameters( will be Learned by fit function)
        
    def fit(self,train_masks): 
        """
        The goal of the fit() function is to learn model parameters. 
        Input:
            * self: the class itself, which includes hyper-parameters of the model
            * masks: A binary 0/1 numpy array matrix of size n by m for training set.

        Output:
            * This function does not return any value. 

        """
        pass # replace by your code.
        
    def predict(self, masks):
        """
        Input:
            * self: the class itself, which includes the model parameters learned by the fit() function.
            * masks: A binary 0/1 numpy array matrix of size n by m for test set.
        Output:
            * This function returns the predicted ratings for the test set. The output is an n by m matrix; all elements that do not belong to the test set (mask = 0) are assigned the value 0. 

        """
        pass # replace by your code. 

"""
Matrix Completion approach. You need to implement fit and predict functions. 
"""        
class MC(Recommender):
    def __init__(self,ratings, tau, delta, max_iters,epsilon):
        Recommender.__init__(self, ratings) 
        '''
        model hyper-parameters 
        '''
        self.delta = delta # step size hyper-parameter for the SVT matrix completion algorithm.
        self.epsilon = epsilon  # the stopping criteria parameter
        self.max_iters = max_iters # max_iters is maximum iteration count
        self.tau = tau  
        '''
        model parameters (will be learned by fit function)
        '''
        self.U = None
        self.S = None 
        self.V = None 

        
    def fit(self, train_masks):
        """
        The goal of the fit() function is to learn model parameters. 
        Input:
            * self: the class itself, which includes hyper-parameters of the model
            * masks: A binary 0/1 numpy array matrix of size n by m for training set.

        Output:
            * This function does not return any value. 

        """
        pass # replace by your code.
    
    def predict(self, masks):
        """
        Input:
            * self: the class itself, which includes the model parameters learned by the fit() function.
            * masks: A binary 0/1 numpy array matrix of size n by m for test set.
        Output:
            * This function returns the predicted ratings for the test set. The output is an n by m matrix; all elements that do not belong to the test set (mask = 0) are assigned the value 0. 

        """
        pass # replace by your code. 



"""
Matrix Factorization approach. You need to implement fit and predict functions. 
""" 
class MF(Recommender):
    def __init__(self,ratings,  max_iters, epsilon, lambda_1, lambda_2, dim):
        Recommender.__init__(self, ratings) 
        '''
        model hyper-parameters 
        '''
        self.epsilon = epsilon # the stopping criteria parameter
        self.max_iters = max_iters # max_iters is maximum iteration count
        self.lambda_1 = lambda_1 # hyperparameters for sparsity regularization for U
        self.lambda_2 = lambda_2 # hyperparameters for sparsity regularization for V
        self.dim = dim # Number of columns for U and V 
        '''
        model parameters (will be learned by fit function)
        '''
        self.n_user, self.n_item = self.ratings.shape
        self.U = np.random.randn(self.n_user, self.dim)
        self.V = np.random.randn(self.n_item, self.dim)
        
    def fit(self,train_masks): 
        """
        The goal of the fit() function is to learn model parameters. 
        Input:
            * self: the class itself, which includes hyper-parameters of the model
            * masks: A binary 0/1 numpy array matrix of size n by m for training set.

        Output:
            * This function does not return any value. 

        """
        pass # replace by your code.
    
    def predict(self, masks):
        """
        Input:
            * self: the class itself, which includes the model parameters learned by the fit() function.
            * masks: A binary 0/1 numpy array matrix of size n by m for test set.
        Output:
            * This function returns the predicted ratings for the test set. The output is an n by m matrix; all elements that do not belong to the test set (mask = 0) are assigned the value 0. 

        """
        pass # replace by your code. 


