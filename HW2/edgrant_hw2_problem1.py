import numpy as np
import matplotlib.pyplot as plt

class HW2EKF:
    def __init__(self, x_0_mean, x_0_var, a_0_mean, a_0_var, a_gt, n_obs):

        self.n_obs = n_obs
        self.a_gt  = a_gt # ground truth coefficient
        
        self.x_0_mean = x_0_mean
        self.x_0_var  = x_0_var
        self.a_0_mean = a_0_mean
        self.a_0_var  = a_0_var

        self.mean  = np.zeros((self.n_obs + 1, 2))
        self.var_x = np.zeros((self.n_obs + 1, ))
        self.var_a = np.zeros((self.n_obs + 1, ))
        
        self.rng = np.random.default_rng(1998)

        self.epsilon_var = 1
        self.nu_var = 0.5
        self.a_noise_var = 0.5

        self.x_gt, self.D = self.generateDataset(a_gt, x_0_mean)

    def getGaussian(self, mean, var, shape = (), useSeed = False):
        '''
        Returns a sample from a Gaussian distribution
        '''
        if useSeed:
            return self.rng.normal(mean, np.sqrt(var), shape)
        else:
            return np.random.normal(mean, np.sqrt(var), shape)

    def generateDataset(self, a_gt, mu_0):
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
        # RNG for Gaussian Sampling
        epsilon = self.getGaussian(0, self.epsilon_var, (self.n_obs, ))
        nu      = self.getGaussian(0,      self.nu_var, (self.n_obs, ))

        # states over n_obs + 1 timesteps
        x = np.zeros((n_obs + 1, ))
        x[0] = mu_0 # initial

        for i in range(n_obs):
            # compute new mean
            x[i+1] = a_gt * x[i] + epsilon[i]

        y = np.sqrt(x[1:]**2 + 1) + nu

        return x, y


    def propagateDynamics(self, mean_k, Cov_k):
        '''
        Propagates the dynamics forward one timestep

                Parameters:
                        mean_k ((2,1) np.array): mean of state at timestep k | k
                        a_k             (float): coefficient at timestep k | k

                Returns:
                        mean_kp ((2,1) np.array): mean of state at timestep k+1 | k
                        Cov_kp  ((2,2) np.array): covariance of state at timestep k+1 | k
        '''
        mu_x = mean_k[0]
        mu_a = mean_k[1]

        mean_kp = np.array([[(mu_a * mu_x), mu_a]])

        # Jacobian
        A = np.array([[mu_a, mu_x],
                      [   0,    1]])

        Cov_kp = A @ Cov_k @ A.T + np.diag([[self.epsilon_var, self.a_noise_var]])

        return mean_kp, Cov_kp

    def incorporateObservation(self, mean_kp, Cov_kp, y_kp1):
        '''
        Incorporates the observation at timestep k+1

                Parameters:
                        mean_kp ((2,1) np.array): mean of state at timestep k+1 | k
                        Cov_kp  ((2,2) np.array): covariance of state at timestep k+1 | k
                        y_kp1            (float): observation at timestep k+1

                Returns:
                        mean_kf ((2,1) np.array): mean of state at timestep k+1 | k+1
                        Cov_kf  ((2,2) np.array): covariance of state at timestep k+1 | k+1
        '''
        mu_x = mean_kp[0]
        mu_a = mean_kp[1]

        # Jacobian
        C = np.array([[mu_x / np.sqrt(mu_x**2 + 1), 0]])

        # Kalman Gain
        K = Cov_kp @ C.T / (C @ Cov_kp @ C.T + self.nu_var)

        # Innovation
        innovation = y_kp1 - np.sqrt(mu_x**2 + 1)

        # Update
        mean_kf = mean_kp + (K * innovation).ravel()
        Cov_kf = (np.eye(2) - K @ C) @ Cov_kp

        return mean_kf, Cov_kf

    def runEKF(self):
        '''
        Runs the EKF on the dataset
        '''
        self.mean[0] = [self.x_0_mean, a_0_mean]
        self.var_x[0] = self.x_0_var
        self.var_a[0] = self.a_0_var

        Cov = np.array([[self.x_0_var, 0],
                        [0, self.a_0_var]])

        for k in range(self.n_obs):

            # Propagate Dynamics
            self.mean[k+1], Cov = self.propagateDynamics(self.mean[k], Cov)

            # Incorporate Observation
            self.mean[k+1], Cov = self.incorporateObservation(self.mean[k+1], Cov, self.D[k])

            self.var_x[k+1] = Cov[0,0]
            self.var_a[k+1] = Cov[1,1]

        return self.mean, self.var_x, self.var_a

    def plotResults(self):
        '''
        Plots the results of the EKF
        '''

        a_gt_array = np.ones(self.n_obs + 1) * self.a_gt
        sigma = np.sqrt(self.var_a)
        a_minus_sigma = self.mean[:,1] - sigma
        a_plus_sigma  = self.mean[:,1] + sigma


        fig, ax = plt.subplots(1, 2, figsize=(12, 6))

        ax[0].plot(self.x_gt, label='x ground truth')
        ax[0].plot(self.mean[:,0], label='x estimate')
        ax[0].plot(self.D, label='y')
        ax[0].set_title('State and Observation')
        ax[0].legend()

        ax[1].plot(a_gt_array, label='a (ground truth)')
        ax[1].plot(self.mean[:,1], label='a (estimation)')
        ax[1].plot(a_minus_sigma, label='a - sigma')
        ax[1].plot(a_plus_sigma, label='a + sigma')
        ax[1].set_title('Coefficient')
        ax[1].legend()

        plt.show()


##############################
###    Execute the code    ###
##############################

n_obs       = 250
a_gt        = -1
x_0_mean    = 1
x_0_var     = 2
a_0_mean    = -15
a_0_var     = 0.5

ekf = HW2EKF(x_0_mean, x_0_var, a_0_mean, a_0_var, a_gt, n_obs)
mean, var_x, var_a = ekf.runEKF()

np.set_printoptions(precision=3)
# print(f'epsilon:\n{epsilon}')
# print(f'x:\n{x}')
# print(f'y:\n{y}')
print(f'mean:\n{mean}')

ekf.plotResults()