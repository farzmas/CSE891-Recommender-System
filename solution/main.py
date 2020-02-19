from parameter_parser import parameter_parser
from utils import tab_printer, RMSE
from recommender import MC, MI, MF
from data_loader import load_data
from tqdm import tqdm
import numpy as np
from texttable import Texttable
def main():
    # read arguments
    args = parameter_parser() # read argument and creat an argparse object
    tab_printer(args)
    rmse_MI = list()
    rmse_MC = list()
    rmse_MF = list()
    for it in tqdm(range(10),'Iteration:'):
        # loading data
        ratings, train_masks, test_masks = load_data(args.input_file, 
                                                     args.test_size, random_seed = it)
        # Mean imputation model  
        mean_imp = MI(ratings)
        mean_imp.fit(train_masks)
        ratings_hat = mean_imp.predict(test_masks)
        rmse_MI.append(RMSE(ratings, ratings_hat, test_masks))
        
        # Matrix  compelition model  
        matrix_comp = MC(ratings, tau = args.tau, delta = args.delta,
                         max_iters= args.max_iters, epsilon = args.epsilon)
        matrix_comp.fit(train_masks)
        ratings_hat = matrix_comp.predict(test_masks)
        rmse_MC.append(RMSE(ratings, ratings_hat, test_masks)) 
        
        # Matrix Factorization 
        matrix_factor = MF(ratings, dim = args.dim ,lambda_1 = args.lambda_1, lambda_2 = args.lambda_2,
                         max_iters= args.max_iters, epsilon = args.epsilon)
        matrix_factor.fit(train_masks)
        ratings_hat = matrix_factor.predict(test_masks)
        rmse_MF.append(RMSE(ratings, ratings_hat, test_masks)) 
        
    print('Mean Imputation RMSE: %f +/- %f' %(round(np.mean(rmse_MI),4)
                                  ,round(np.var(rmse_MI),4))) 
    print('Matrix Completion RMSE: %f +/- %f' %(round(np.mean(rmse_MC),4)
                                ,round(np.var(rmse_MC),4)))     
    print('Matrix Factorization RMSE: %f +/- %f' %(round(np.mean(rmse_MF),4)
                                ,round(np.var(rmse_MF),4)))   
if __name__ =="__main__":
    main()