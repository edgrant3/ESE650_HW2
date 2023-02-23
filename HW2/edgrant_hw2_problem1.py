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
        self.a_noise_var = 0.01

        self.x_gt, self.D = self.generateDataset(a_gt, x_0_mean)


    def getGaussian(self, mean, var, shape = (), useSeed = False):
        if useSeed:
            return self.rng.normal(mean, np.sqrt(var), shape)
        else:
            return np.random.normal(mean, np.sqrt(var), shape)

    def generateDataset(self, a_gt, mu_0):
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


    def propagateDynamics(self, mean_k_k, Cov_k_k):
        mu_x = mean_k_k[0]
        mu_a = mean_k_k[1]

        mean_k1_k = np.array([(mu_a * mu_x), mu_a])

        # Jacobian
        A = np.array([[mu_a, mu_x],
                      [   0,    1]])

        Cov_k1_k = A @ Cov_k_k @ A.T + np.diag([self.epsilon_var, self.a_noise_var])

        return mean_k1_k, Cov_k1_k

    def incorporateObservation(self, mean_k1_k, Cov_k1_k, y_k1):
        mu_x = mean_k1_k[0]

        # Jacobian
        C = np.array([mu_x / np.sqrt(mu_x**2 + 1), 0]).reshape(1,2)

        # Kalman Gain
        K = Cov_k1_k @ C.T @ np.linalg.inv(C @ Cov_k1_k @ C.T + self.nu_var)

        # Innovation
        innovation = y_k1 - np.sqrt(mu_x**2 + 1)
        # Update
        mean_k1_k1 = mean_k1_k + (K * innovation).ravel()
        Cov_k1_k1 = (np.eye(2,2) - K @ C) @ Cov_k1_k

        return mean_k1_k1, Cov_k1_k1

    def runEKF(self):

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

        fig, ax = plt.subplots(1, 2, figsize=(18, 6))

        fig.suptitle('EKF Results')

        ax[0].plot(self.x_gt, label='x ground truth')
        ax[0].plot(self.mean[:,0], label='x estimate')
        ax[0].plot(self.D, label='Dataset (observations)')
        ax[0].set_title('State and Observation')
        ax[0].set_xlabel("Timesteps")
        ax[0].legend()

        ax[1].plot(a_gt_array, label='a (ground truth)', color='black', linestyle='--')
        ax[1].plot(self.mean[:,1], label='a (estimation)', color='red')
        ax[1].fill_between(range(self.n_obs + 1), a_minus_sigma, a_plus_sigma, color='gray', alpha=.2, label='a +/- sigma')
        # ax[1].plot(a_minus_sigma, label='a - sigma')
        # ax[1].plot(a_plus_sigma, label='a + sigma')
        ax[1].set_title('Coefficient (a)')
        ax[1].set_xlabel("Timesteps")
        ax[1].legend(loc = 'lower right')

        plt.show()


##############################
###    Execute the code    ###
##############################

n_obs       = 100
a_gt        = -1
x_0_mean    = 1
x_0_var     = 2
a_0_mean    = -5
a_0_var     = 0.5

ekf = HW2EKF(x_0_mean, x_0_var, a_0_mean, a_0_var, a_gt, n_obs)
mean, var_x, var_a = ekf.runEKF()

ekf.plotResults()