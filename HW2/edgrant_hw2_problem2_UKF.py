import numpy as np
import scipy
import scipy.spatial.transform
from scipy import io

import matplotlib.pyplot as plt

# Load IMU data (code from HW2 handout)
data_num = 1
# imu: dict of data with keys 'ts', 'vals' and respective values:
# (1x5645) array of timestamps | (6x5645) array of measurements
imu = io.loadmat('imu/imuRaw' + str(data_num) + '.mat') 
accel = imu['vals'][0:3, :] # (3, 5645) array of ADC ints
gyro  = imu['vals'][3:6, :] # (3, 5645) array of ADC ints
T = np.shape(imu['ts'])[1]  # number of timesteps = 5645

# Load "ground truth" VICON data (COMMENT OUT FOR AUTOGRADER SUBMISSION)
# dict with keys 'rots' and 'ts' := 
# (3, 3, 5561) array of rotations | (1x5561) array of timestamps
vicon = io.loadmat('vicon/viconRot' + str(data_num) + '.mat')


# Helper functions for converting ADC readings to physical units
def ADCtoAccel(adc):
    '''
    Converts ADC readings from accelerometer to m/s^2
    Input:  adc - (int np.array shape (3, )) ADC reading
    Output: acc - (float np.array shape (3, )) acceleration in m/s^2
    '''
    bias        = np.array([0, 0, 0])       # (mV)
    sensitivity = np.array([500, 500, 500]) # (mV/grav)
    return (adc - bias) * 3300 / (1023 * sensitivity) * 9.81

def ADCtoGyro(adc):
    '''
    Converts ADC readings from gyroscope to rad/s
    Input:  adc - (int np.array shape (3, )) ADC reading
    Output: gyr - (float np.array shape (3, )) angular velocity in rad/s
    '''
    bias        = np.array([0, 0, 0])       # (mV)
    sensitivity = np.array([250, 250, 250]) # (mV/(rad/sec))
    return (adc - bias) * 3300 / (1023 * sensitivity)


''' CAUTION MOVING FORWARD:
(1) The orientation of the IMU need not be the same as the orientation of the Vicon
coordinate frame. Plot all quantities in the arrays accel, gyro and vicon rotation
matrices to make sure you get this right. Do not proceed to implementing the filter if
you are not convinced your solution for this part is correct.

(2) The acceleration ax and ay is flipped in sign due to device design. A positive
acceleration in body-frame will result in a negative number reported by the IMU.
See the IMU manual for more insight.
'''