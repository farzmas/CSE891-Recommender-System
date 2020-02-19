import numpy as np

"""
Base Recommender class. This is the base class for Mean Imputation(MI), Matrix campletion(MC), and Matrix Factorization (MF) classes. You do not need to modify this class.
"""
class Recommender(object):
    def __init__(self,ratings):
        self.ratings = ratings # actual rating matrix 
    
    def fit(self, train_masks): #
        """
        Model fitting function,
         * input is train_mask
         * this funciton does not return any value it stores the estimated ratings in self.ratings_hat variable. 
        """
        pass
    
    def predict(self, masks): 
        """
        prediction function return the estimated ratings for the given input mask
        """
        return masks
    
    def SVD(self, X): 
        """
         SVD decomposition function         
        """
        return np.linalg.svd(X, full_matrices=False)



        
"""
Mean Imputation approach
"""
class MI(Recommender): 
    def __init__(self,ratings):
        Recommender.__init__(self, ratings)         
        self.rating_avg = None # The Mean Imputation model parameters(will be learned by fit function)
    
    def fit(self,masks):  
        self.n_user, self.n_item = self.ratings.shape
        self.rate_avg = np.mean(self.ratings*masks,0)
        
    def predict(self,masks):
        return np.array([self.rate_avg,]*self.n_user)*(masks)
        

"""
Matrix Completion approach
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
        
        
    def fit(self, masks):
        self.n_user, self.n_item = self.ratings.shape
        if self.delta is None:
            self.delta = 1.2 *( self.n_user* self.n_item)/ np.sum(masks)
        if self.tau is None:
            self.tau = 2.5 * np.sum(self.n_user + self.n_item)
        Y = np.zeros((self.n_user, self.n_item))
        for it in range(self.max_iters):
            self.U, self.S, self.V = self.SVD(Y) #
            self.S = np.maximum(self.S - self.tau, 0)
            X = np.linalg.multi_dot([self.U, np.diag(self.S), self.V])
            Y += self.delta * masks * (self.ratings - X)
            error =  np.linalg.norm(masks * (X - self.ratings)) / np.linalg.norm(masks * self.ratings)
            if error < self.epsilon:
                break
        
    def predict(self,masks):
        X = np.linalg.multi_dot([self.U, np.diag(self.S), self.V])
        return X*(masks) 

"""
Matrix Factorization approach
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
        self.U = None
        self.V = None 
    def fit(self, masks):
        self.n_user, self.n_item = self.ratings.shape
        self.U = np.random.randn(self.n_user, self.dim)
        self.V = np.random.randn(self.n_item, self.dim)  
        for it in range(self.max_iters):
            for i in range(self.n_user):
                left_ = np.zeros((1,self.dim))
                right_ = self.lambda_1*np.identity(self.dim)
                for j in range(self.n_item):
                    if masks[i][j] == 1:
                        v_t = self.V[j].reshape(1,self.dim)
                        left_ = left_ + self.ratings[i][j]*v_t
                        right_ = right_ + v_t.transpose().dot(v_t)
                u_t = left_.dot(np.linalg.inv(right_))
                for j in range(self.dim):
                    self.U[i][j] = u_t[0][j]

            for j in range(self.n_item):
                left_ = np.zeros((1,self.dim))
                right_ = self.lambda_1*np.identity(self.dim)
                for i in range(self.n_user):
                    if masks[i][j] == 1:
                        u_t = self.U[i].reshape(1,self.dim)
                        left_ = left_ + self.ratings[i][j]*u_t
                        right_ = right_ + u_t.transpose().dot(u_t)
                v_t = left_.dot(np.linalg.inv(right_))
                for i in range(self.dim):
                    self.V[j][i] = v_t[0][i]


    def predict(self,masks):
        X = np.dot(self.U, self.V.T)
        return X*(masks) 
        


    