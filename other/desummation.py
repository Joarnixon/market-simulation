from typing import Any
import numpy as np
import random as rd
import scipy.spatial as sp
import optuna
from functools import partial
from scipy.optimize import nnls

class Desummation():
    '''
    If you want to experiment with bayesian optimization technique for weights searching
    or if you find it more suitable for you problem use frobenius=False
    in other cases frobenius=True is the best choice.
    '''
    def __init__(self, frobenius=True):
        self.basis = None
        self.frobenius = frobenius
        self.weights = None
        self.error = None

        if frobenius == True:
            self.optimizer : Weights = None
        else:
            self.optimizer : Weights_old = None
        
    def fit(self, A, k=None, positive=False, **kwargs):
        '''
        Arguments:
            A - matrix to fit
            k - how many random matrices will be generated
            positive - Whether or not to solve in positive.
            
            **kwargs - additional keyword arguments

        Keyword Args:

            For creating random matrices:
                distribution (str): The desired distribution of elements in the matrices. 
                Supported distributions: 'normal', 'exponential', 'uniform', 'binomial', 'bernoulli', 'integer'.

                mean (float): The mean value(s) for the normal distribution. 
                variance (float): The variance value(s) for the normal distribution. 
                scale (float): The scale parameter for the exponential distribution.
                low (float): The lower bound for the uniform distribution.
                high (float): The upper bound for the uniform distribution.
                n (int): The number of trials for the binomial distribution.
                p (float): The probability of success for the binomial and Bernoulli distributions.
                low (int): The lower bound for the integer distribution.
                high (int): The upper bound for the integer distribution.

            For finding weights:
                n_trials (int): Number of optimization trials
                distance (str): Distance metric to use for calculating the error. Default is 'euclidean'.
                Supported distance: 'chebyshev', 'euclidean', 'cosine', 'cityblock', 'canberra', 'correlation', 'braycurtis'.
                Can be added new distance metrics inside class instance if you know what you are doing.
        '''
        A = np.array(A)
        if len(A.shape) == 1:
            A = A.reshape(A.shape[0], 1)
        if k is None:
            k = len(A)
        B = RandomMatrices(A.shape)
        B.add(k, **kwargs)
        self.basis = B.matrices

        if self.frobenius == True:
            self.optimizer = Weights(**kwargs)
        else:
            self.optimizer = Weights_old(**kwargs)
        self.optimizer.fit(A, B.matrices, positive)
        self.weights = self.optimizer.weights
        error = self.optimizer.error
        if isinstance(error, list):
            self.error = error[-1]
        else:
            self.error = error
    
    def predict(self, A, positive=False):
        '''
        Fits to a given matrix A without creating new random matrices
        '''
        predicted = self.optimizer.fit_predict(A, self.basis, positive)
        self.weights = self.optimizer.weights
        self.error = self.optimizer.error
        return predicted
    
    def fit_predict(self, A, k=None, positive=False, **kwargs):
        '''
        Fits and then predicts to a given matrix A with creating new random matrices
        
        Arguments:
            A - matrix to fit
            k - how many random matrices will be generated
            positive - Whether or not to solve in positive.
            **kwargs - additional keyword arguments

        Keyword Args:

            For creating random matrices:
                distribution (str): The desired distribution of elements in the matrices. 
                Supported distributions: 'normal', 'exponential', 'uniform', 'binomial', 'bernoulli', 'integer'.

                mean (float): The mean value(s) for the normal distribution. 
                variance (float): The variance value(s) for the normal distribution. 
                scale (float): The scale parameter for the exponential distribution.
                low (float): The lower bound for the uniform distribution.
                high (float): The upper bound for the uniform distribution.
                n (int): The number of trials for the binomial distribution.
                p (float): The probability of success for the binomial and Bernoulli distributions.
                low (int): The lower bound for the integer distribution.
                high (int): The upper bound for the integer distribution.

            For finding weights:
                n_trials (int): Number of optimization trials
                distance (str): Distance metric to use for calculating the error. Default is 'euclidean'.
                Supported distance: 'chebyshev', 'euclidean', 'cosine', 'cityblock', 'canberra', 'correlation', 'braycurtis'.
                Can be added new distance metrics inside class instance if you know what you are doing.
        '''
        if k is None:
            k = len(A)
        self.fit(A, k, positive, **kwargs)
        return self.predict(A, positive)
            


class RandomMatrices:
    """
    Attributes:
        matrices (List): A list to store the generated matrices.
        shape (tuple or integer): Shape same as matrix to be converted

    Methods:
        add(self, amount: int, distribution: str): Appends certain amount of random matrices to the class object.
        __init__(self, distribution: str): Initializes the RandomMatrices object.
        __generate(self, distribution: str) -> List: Inner method, generates the random matrices based on the given distribution.
    """
    def __init__(self, shape, matrices=[]):
        self.matrices = np.array(matrices)
        if isinstance(shape, int):
            self.shape = (shape, shape)
        elif isinstance(shape, tuple):
            self.shape = shape
        else:
            raise ValueError("Invalid shape provided. Please pass a tuple (rows, columns) or an integer for square matrix")
        if matrices:
            if not all(matrix.shape == self.shape for matrix in matrices):
                raise ValueError("Invalid matrices provided. Please pass with equal shapes.")
            self.matrices = matrices
    
    def add(self, amount: int, **kwargs) -> None: 
        """
        Appends certain amount of random matrices to the class object.

        Args:
            amount (int): The number of matrices.
            **kwargs: Additional keyword arguments for the distribution parameters.

        Keyword Args:
            distribution (str): The desired distribution of elements in the matrices. 
            Supported distributions: 'normal', 'exponential', 'uniform', 'binomial', 'bernoulli', 'integer'.

            mean (float): The mean value(s) for the normal distribution. 
            variance (float): The variance value(s) for the normal distribution. 
            scale (float): The scale parameter for the exponential distribution.
            low (float): The lower bound for the uniform distribution.
            high (float): The upper bound for the uniform distribution.
            n (int): The number of trials for the binomial distribution.
            p (float): The probability of success for the binomial and Bernoulli distributions.
            low (int): The lower bound for the integer distribution.
            high (int): The upper bound for the integer distribution.
        """
        if amount > 0:
            self.matrices = self.__generate(amount, **kwargs)
        else:
            raise ValueError("Amount of matrices should be strictly positive")

    def __generate(self, amount: int, **kwargs) -> list:
        matrices = list(self.matrices)
        distribution = kwargs.get('distribution', 'normal')
        if distribution == 'normal':
            mean = kwargs.get('mean', 0)
            variance = kwargs.get('variance', 1)
            for i in range(amount):
                matrix = np.random.normal(mean, variance, self.shape)
                matrices.append(matrix)
        elif distribution == 'exponential':
            scale = kwargs.get('scale', 1.0)
            for i in range(amount):
                matrix = np.random.exponential(scale, self.shape)
                matrices.append(matrix)
        elif distribution == 'uniform':
            low = kwargs.get('low', 0.0)
            high = kwargs.get('high', 1.0)
            for i in range(amount):
                matrix = np.random.uniform(low, high, self.shape)
                matrices.append(matrix)
        elif distribution == 'binomial':
            n = kwargs.get('n', 1)
            p = kwargs.get('p', 0.5)
            for i in range(amount):
                matrix = np.random.binomial(n, p, self.shape)
                matrices.append(matrix)
        elif distribution == 'bernoulli':
            p = kwargs.get('p', 0.5)
            for i in range(amount):
                matrix = np.random.binomial(1, p, self.shape)
                matrices.append(matrix)
        elif distribution == 'integer':
            low = kwargs.get('low', 0)
            high = kwargs.get('high', 10)
            for i in range(amount):
                matrix = np.random.randint(low, high, self.shape)
                matrices.append(matrix)
        return matrices
    
    def __str__(self):
        return str(self.matrices)
    
    def __getitem__(self, indx):
        return self.matrices[indx]
    
    def clear(self):
        self.matrices = np.array([])


class Weights:
    """
    Class for decomposing A into a weighted sum of B_matrices.

    Parameters:
    - **kwargs: Additional keyword arguments.
        - random_state(int): Random seed for reproducibility.
    """
    def __init__(self, **kwargs) -> None:
        self.random_state = kwargs.get('random_state', rd.randint(0, 100000))
        self.error = 0
        self.weights = None
    
    def loss(self, basis, target, distance='fro'):
        return np.linalg.norm(target - np.tensordot(self.weights, basis, axes=1), ord=distance)

    def fit(self, A, B_matrices, positive):
        """
        Fit the weights to decompose A into a weighted sum of B_matrices.

        Parameters:
        - A (ndarray): Matrix to decompose.
        - B_matrices (list): List of matrices which will be used to decompose.
        - positive (bool): Whether or not to solve in positive.
        """
        target = np.array(A)
        if len(target.shape) == 1:
            target = target.reshape(target.shape[0], 1)
        shape = target.shape
        basis = np.array(B_matrices)
        k = len(basis)

        C = np.zeros((shape[0] * shape[1], k))

        for i in range(k):
            C[:, i] = basis[i].flatten()
        p = np.array(target).flatten()
        # Use the least squares method to solve the system
        if positive is False:
            x, residuals, rank, singular_values = np.linalg.lstsq(C, p, rcond=None)
        else:
            x, e = nnls(C, p, maxiter=5*C.shape[1])
        # The solution vector x contains the values of w1, w2, w3, ..., wk
        w = x[:k]
        self.weights = w
        self.error = self.loss(basis=basis, target=target)

    def predict(self, B_matrices):
        """
        Parameters:
        - B_matrices (list): List of matrices which will be used to decompose.

        Returns:
        - weighted_sum (ndarray): Weighted sum of B_matrices.
        """
        B_matrices = np.array(B_matrices)
        assert self.weights is not None, 'Weights error, must be fitted before predict'
        weighted_sum = np.tensordot(self.weights, B_matrices, axes=1)
        return weighted_sum

    def fit_predict(self, A, B_matrices, positive):
        """
        Fit the weights and returns the weighted sum of B_matrices.

        Parameters:
        - A (ndarray): Matrix to decompose.
        - B_matrices (list): List of matrices which will be used to decompose.
        - positive (bool): Whether or not to solve in positive.
        """
        self.fit(A, B_matrices, positive)
        return self.predict(B_matrices)
    
    def __str__(self):
        return str(self.weights)
    
    def __getitem__(self, indx):
        return self.weights[indx]
    

class Weights_old:
    """
    Class for decomposing A into a weighted sum of B_matrices.

    Parameters:
    - random_state (int): Random seed for reproducibility.
    - **kwargs: Additional keyword arguments.
        - n_trials (int): Number of optimization trials
        - distance (str): Distance metric to use for calculating the error. Default is 'euclidean'.
        - Supported distance: 'chebyshev', 'euclidean', 'cosine', 'cityblock', 'canberra', 'correlation', 'braycurtis'.
        - Can be added new distance metrics inside class instance if you know what you are doing.

    Attributes:
    - study: Optuna study object.
    - weights (list): List of weights for the B_matrices.
    - random_state (int): Random seed for reproducibility.
    - n_trials (int): Number of optimization trials.
    - errors (list): List of errors during optimization.
    - distance (str): Distance metric to use for calculating the error.
    - supported (list): List of supported distance metrics.
    - min: Minimum value in A.
    - max: Maximum value in A.
    """
    def __init__(self, random_state=rd.randint(0, 10000), **kwargs) -> None:
        self.random_state = random_state
        self.error = []
        self.supported_scipy = ['chebyshev', 'euclidean', 'cosine', 'cityblock', 'canberra', 'correlation', 'braycurtis']
        self.supported_numpy = ['fro', 'nuc', np.inf, -np.inf, 1, 2, -1, -2]
        self.distance = kwargs.get('distance', 'fro')
        self.n_trials = kwargs.get('n_trials', 50)
        self.distance_library = None
        self.min = None
        self.max = None
        self.study = None
        self.weights = None

    def __distance(self, A, B_matrices) -> float:
        """
        Calculate the distance between A and the weighted sum of B_matrices.

        Parameters:
        - A (ndarray): True values.
        - B_matrices (list): List of predicted matrices.

        Returns:
        - distance (float): Distance between A and the weighted sum of B_matrices.
        """
        distance = self.distance
        if distance in self.supported_numpy:
            self.distance_library = 'numpy'
        elif distance in self.supported_scipy:
            self.distance_library = 'scipy'
        else:
            raise ValueError("Distance metric is not supported. Append it to class atribute or choose from Numpy: 'fro', 'nuc', 'inf', '-inf', '0', '1', '2', '-1', '-2', Scipy: 'chebyshev', 'euclidean', 'cosine', 'cityblock', 'canberra', 'correlation', 'braycurtis'")
            return False
        
        B_matrices = np.array(B_matrices)
        weighted_sum = sum(weight * B for weight, B in zip(self.weights, B_matrices))
        
        A = np.array(A)
        if len(A.shape) == 1:
            A = A.reshape(A.shape[0], 1)

        if self.distance_library == 'numpy':
            return np.linalg.norm(A - weighted_sum, ord=distance)
        else:
            return np.sum(sp.distance.cdist(A, weighted_sum, distance))

    def __objective(self, trial, A, B_matrices):
        """
        Objective function for optimization.

        Parameters:
        - trial: Optuna trial object.
        - A (ndarray): True values.
        - B_matrices (list): List of predicted matrices.

        Returns:
        - score (float): Error score.
        """
        self.weights = [trial.suggest_float(f"weight{n}", -self.max, self.max) for n in range(len(B_matrices))]
        score = self.__distance(A=A, B_matrices=B_matrices)
        self.error.append(score)
        return score

    def fit(self, A, B_matrices):
        """
        Fit the weights to decompose A into a weighted sum of B_matrices.

        Parameters:
        - A (ndarray): Matrix to decompose.
        - B_matrices (list): List of matrices which will be used to decompose.
        """
        self.min = np.min(A)
        self.max = np.max(A)
        optuna.logging.set_verbosity(optuna.logging.ERROR)
        sampler = optuna.samplers.CmaEsSampler(seed=self.random_state)
        pruner = optuna.pruners.HyperbandPruner()
        self.study = optuna.create_study(sampler=sampler, pruner=pruner, study_name="OptunaWeights", direction='minimize')
        objective_partial = partial(self.__objective, A=A, B_matrices=B_matrices)
        self.study.optimize(objective_partial, n_trials=self.n_trials)
        self.weights = [self.study.best_params[f"weight{n}"] for n in range(len(B_matrices))]


    def predict(self, B_matrices):
        """
        Parameters:
        - B_matrices (list): List of matrices which will be used to decompose.

        Returns:
        - weighted_sum (ndarray): Weighted sum of B_matrices.
        """
        B_matrices = np.array(B_matrices)
        assert self.weights is not None, 'Weights error, must be fitted before predict'
        weighted_sum = np.tensordot(self.weights, B_matrices, axes=1)
        return weighted_sum

    def fit_predict(self, A, B_matrices):
        """
        Fit the weights and returns the weighted sum of B_matrices.

        Parameters:
        - A (ndarray): Matrix to decompose.
        - B_matrices (list): List of matrices which will be used to decompose.
        """
        self.fit(A, B_matrices)
        return self.predict(B_matrices)



