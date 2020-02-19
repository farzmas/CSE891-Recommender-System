<<<<<<< HEAD
# Project 1: Recommender system 

This is the first of two mini projects in this class. Each mini project accounts for 10% fo your final grade. The project requires implementing and evaluating various methods for a joke recommender system. The project due date is Sunday, Oct 20, 2019 (before midnight).

### Project files
First, you need to download the Project1.zip from the D2L class website. After unpacking, you should find a folder named skeleton, which contains the following files:
* main.py
* recommender.py
* parameter_parser.py
* utils.py
* data_loader.py

There should also be a directory named *data* that contains the  jester.csv file. Your task is to implement some of the Python functions given by the skeleton code in data_loader.py, recommender.py, and utils.py. More details are given below. 

### Data:
The dataset in the *data* folder is a subset of the Jester Online dataset for joke recommendation. The original data contains 4.1 Million continuous-valued ratings (between -10.00 to +10.00) of 100 jokes from 73,421 users. The data was collected between April 1999 - May 2003. You can access to the original dataset from following link: http://goldberg.berkeley.edu/jester-data/. 
 Each row in the jester.csv file corresponds to ratings of a user. There are 100 columns(comma seperated) in this file each column corresponds to a joke and Ratings are real values ranging from -10.00 to +10.00.

### How to run?
The skeleton program is written in python 3.7. Inorder to be able to run the skeleton you need to install following packages using pip or conda command:
- tqdm
- numpy 
- texttable
- argparse

If everything is installed correctly you should be able to run the following command without generating any error:
```shell-script
$ python main.py --help
usage: main.py [-h] [--input-file [INPUT_FILE]] [--tau TAU] [--delta DELTA] [--epsilon EPSILON] [--lambda-1 LAMBDA_1]
               [--lambda-2 LAMBDA_2] [--max_iters MAX_ITERS] [--dim DIM] [--test-size TEST_SIZE] [--file FILE]

Run recommender system algorithms.

optional arguments:
  -h, --help            show this help message and exit
  --input-file [INPUT_FILE]
                        path to input file.
  --tau TAU             parameter tau of the SVT matrix completion algorithm.
  --delta DELTA         step size hyper-parameter for the SVT matrix completion algorithm.
  --epsilon EPSILON     the stopping criteria parameter for SVT matrix completion algorithm. Default is 1e-2
  --lambda-1 LAMBDA_1   hyperparameters for sparsity regularization for U in MF algorithm. Default is 1e-2
  --lambda-2 LAMBDA_2   hyperparameters for sparsity regularization for V in MF algorithm. Default is 1e-2
  --max_iters MAX_ITERS
                        max_iters is maximum iteration count for the MC and MF algorithm. Defualt is 1000.
  --dim DIM             Number of columns for U and V in MF algorithm. Defualt is 10.
  --test-size TEST_SIZE
                        the proportion of the dataset to include in the test split. It should be between 0 and 1. Default is 0.5
  --file FILE
```

## Tasks
### 1. Data Loading and Creation

The data_loader.py file contains the template program for loading the dataset. The file includes a function named *load_data()*, which reads the input data from a file and generates the appropriate training and test sets. You need to modify the load_data() function to complete this task.


#### load_data():
Input:
- input_file: path to input file.
- test_size: a floating point number between 0.0 and 1.0, which represents the proportion of the input data to be used as the test set.
- random_seed:  an integer seed to be used by the random number generator. 

Output:
- Ratings: A numpy array of user ratings of size n(number of users) by m(number of items). Each element(u,b) is the rating of user u for the item b. 
- train_mask:  A binary 0/1 numpy array matrix of size n by m. Element(u,b) is set to 1 if the user-item pair (u,b) belongs to the training set; otherwise, it is set to 0.
- test_mask: A binary 0/1 numpy array matrix of size n by m. Element(u,b) is set to 1 if the user-item pair (u,b) belongs to the test set; otherwise, it is set to 0.

### 2. Recommendation (recommender.py)

This file includes the base recommender class and the skeleton for MI (mean imputation), MC (Matrix completion), and MF( matrix factorization) derived classes. In this file you need to implement fit() and predict() functions for each of the derived classes. All the required  hyper-parameter are defined in the __init__() so you probably do not need to define
any extra hyper-parameter. 

#### fit():
The goal of the fit() function is to learn model parameters. The function takes the training data as input and learns the model parameters. This function does not return any value.

Input:
* self: the class itself, which includes hyper-parameters of the model
* masks: A binary 0/1 numpy array matrix of size n by m for training set.

Output:
* This function does not return any value. 

##### Matrix Completion fit() function:
* for implementation details and algorithm, follow the pseudocode given in Section 5 of
https://epubs.siam.org/doi/pdf/10.1137/080738970
* You can ignore steps 4-8 of the algoirthm 1(above link) in your implementation. 

##### Matrix Factorization fit() function:

Matrix Factorization fit() function use alternating least-square(ALS) algoirthm to solve follwoing objective function:
 $$min_{U,V} ||(R - UV^{\top})\odot M||^2_F + \lambda_1||U||^2_F + \lambda_2 ||V||^2_F $$

The main steps of ALS are as follow 

- Randomly intialize U and V
- Repeat until convergence:
    - Fix V and update U. The update formula for U can be derived by seting gradient of the above objective function with respect to U equal to 0 and solving for U.
    - Fix U and update V. The update formula for U can be derived by seting gradient of the above objective function with respect to V equal to 0 and solving for V.


#### predict():
Input:
* self: the class itself, which includes the model parameters learned by the fit() function.
* masks: A binary 0/1 numpy array matrix of size n by m for test set.

Output:
* This function returns the predicted ratings for the test set. The output is an n by m matrix; all elements that do not belong to the test set (mask = 0) are assigned the value 0. 



### 3.Evaluation (utils.py)

The final task is to implement the RMSE() function in utils.py to compute the root mean square error of the predicted ratings for the test set. Details of the function are given below.

RMSE():

Input:
- ratings:  numpy array matrix of actual user ratings.
- ratings_hat: numpy array matrix of predicted user ratings.
- test_mask:  0/1 test mask matrix. 

Output:
- This function returns the root mean square error (rmse) of the predictions. Note that the rmse should be computed using only those entries whose test_mask is equal to 1.


## Implementation policies

* The project must be done individually, without collaboration with any other students.
* You must implement your methods in Python 3.7 using the provided template program.
* You are prohibited from using any Python libraries for recommender systems (e.g., Python Surprise package) to do this project. 
* You’re expected to complete the project using standard functions in Python and NumPy built in methods such as the NumPy linear algebra functions and Numpy’s random number routines. 
* Please check with the instructor/TA first if you want to use other packages besides those provided by NumPy. 

## Project deliverables
You should turn in following files:
* recommender.py
* data_loader.py
* utils.py
* results.txt: Run the program with the default hyper-parameter settings and report the  results in this file.
=======
# CSE891-Recommender-System
CSE891 Mini Project 1: Recommender System
>>>>>>> 4fc5b964629c27c2d1f40b1156a8b4c68f906c86
