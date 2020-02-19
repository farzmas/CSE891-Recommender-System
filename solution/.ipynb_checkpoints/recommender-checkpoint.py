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
            #print(error)
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
            self.U = np.dot((masks*self.ratings),self.V)/(np.dot(masks,self.V)*np.dot(masks,self.V)+self.lambda_1)
            self.V = np.dot((masks*self.ratings).T,self.U)/(np.dot(masks.T,self.U )*np.dot(masks.T,self.U)+self.lambda_2)
            X = np.dot(self.U, self.V.T)
            error =  np.linalg.norm(masks * (X - self.ratings))/ np.linalg.norm(masks * self.ratings)
            #print(error)
            if error < self.epsilon:
                print('break')
                break
        
    def predict(self,masks):
        X = np.dot(self.U, self.V.T)
        return X*(masks) 
        
# ###################################################
# class MC_old(object):
#     def __init__(self, args, ratings, masks ):
#         self.args = args
#         self.ratings = ratings
#         self.masks = masks
        
#     def fit(self,alg):
#         if alg == 'mean':
#             R_hat = self.mmc()
#         if alg == 'svt':
#             R_hat = self.SVT()
#         return R_hat
            
#     def mmc(self):
#         n_user, n_item = self.ratings.shape
#         rate_avg = np.mean(self.ratings*self.masks,0)
#         X = self.ratings*self.masks + np.array([rate_avg,]*n_user)*(1-self.masks)        
#         return X

#     def SVT(self):
#         n_user, n_item = self.ratings.shape
#         Y = np.zeros((n_user, n_item))
#         if self.args.tau ==0:
#             self.args.tau = 2.5 * np.sum(n_user + n_item)
#             print('tau',self.args.tau)
#         if self.args.delta ==0:
#             self.args.delta = 1.2 *( n_user* n_item)/ np.sum(self.masks)
#             print('delta', self.args.delta)

#         for it in range(self.args.max_iters):
#             #print(it, end =' ')
#             try:
#                 U, S, V = self.SVD(Y) #
#                 S = np.maximum(S - self.args.tau, 0)
#                 X = np.linalg.multi_dot([U, np.diag(S), V])
#                 Y += self.args.delta * self.masks * (self.ratings - X)
#                 error =  np.linalg.norm(self.masks * (X - self.ratings)) / np.linalg.norm(self.masks * self.ratings)
#                 #print(error)
#                 if error < self.args.epsilon:
#                     break
#                 if it%10 ==0:
#                     print('iteration', it , 'error:',error)
#             except:
#                 print(Y)
#                 break
#         return X
    

    