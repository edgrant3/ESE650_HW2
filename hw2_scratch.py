import numpy as np
import scipy

n  = 5
mean = np.arange(n)
sig_pts = np.ones((n, 2*n)) * mean.reshape(n, 1)
sig_pts[-2:,:] = 0
print(sig_pts)