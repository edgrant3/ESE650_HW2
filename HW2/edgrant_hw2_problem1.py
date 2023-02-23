import numpy as np

def generateDataset(n_obs = 100, a_gt = -1, mu_0  = 1):
    '''
    Returns a forward simulation of the dynamics and observation for HW2

            Parameters:
                    n_obs   (int): number of observations in simulation
                    a_gt  (float): ground-truth coefficent in dynamics
                    mu_0  (float): mean of initial state x_0
                    var_0 (float): variance of initial state x_0

            Returns:
                    x ((n_obs + 1, ) np.array): states over n_obs timesteps
                    y ((n_obs, ) np.array): observations from n_obs timesteps
    '''

    # Compute Gaussian noise components for dynamics and observations
    var_epsilon, var_nu = 1, 0.5
    epsilon, nu = None, None

    # RNG for Gaussian Sampling
    rng = np.random.default_rng(2023)
    sameseed = True
    if sameseed:
        # --- params: mean, stdev, shape
        epsilon = rng.normal(0, var_epsilon, (n_obs, ))
        nu = rng.normal(0, np.sqrt(var_nu), (n_obs, ))
    else:
        # --- params: mean, stdev, shape
        epsilon = np.random.normal(0, var_epsilon, (n_obs, ))
        nu = np.random.normal(0, np.sqrt(var_nu), (n_obs, ))

    # states over n_obs + 1 timesteps
    x = np.zeros((n_obs + 1, ))
    x[0] = mu_0 # initial

    for i in range(n_obs):
        # compute new mean
        x[i+1] = a_gt * x[i] + epsilon[i]

    y = np.sqrt(x[1:]**2 + 1) + nu

    return x, y

n_obs = 100
a_gt = -1
mu_0  = 1
var_0 = 2

x, y = generateDataset(n_obs, a_gt, mu_0)
np.set_printoptions(precision=3)
# print(f'epsilon:\n{epsilon}')
print(f'x:\n{x}')
print(f'y:\n{y}')

# --------------
a_0_mean = -10
a_0_var  = 0.5
Cov_0 = np.array([[1, 0],
                  [0, a_0_var]])
mean_0 = np.array([[0],
                   [a_0_mean]])

def propagateDynamics(state, )