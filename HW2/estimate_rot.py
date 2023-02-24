import numpy as np
from scipy import io
# from scipy.spatial import transform
from quaternion import Quaternion
import math
import matplotlib.pyplot as plt

#data files are numbered on the server.
#for exmaple imuRaw1.mat, imuRaw2.mat and so on.
#write a function that takes in an input number (1 through 6)
#reads in the corresponding imu Data, and estimates
#roll pitch and yaw using an unscented kalman filter

def load_imu_data(data_num):
    # imu: dict of data with keys 'ts', 'vals' and respective values:
    # (1x5645) array of timestamps | (6x5645) array of measurements
    imu = io.loadmat('imu/imuRaw' + str(data_num) + '.mat') 
    accel = imu['vals'][0:3, :] # (3, 5645) array of ADC ints
    gyro  = imu['vals'][3:6, :] # (3, 5645) array of ADC ints
    T = np.shape(imu['ts'])[1]  # number of timesteps = 5645
    return accel, gyro, T

def load_vicon_data(data_num):
    # Load "ground truth" VICON data (COMMENT OUT FOR AUTOGRADER SUBMISSION)
    # dict with keys 'rots' and 'ts' := 
    # (3, 3, 5561) array of rotations | (1x5561) array of timestamps
    vicon = io.loadmat('vicon/viconRot' + str(data_num) + '.mat')
    return vicon

# Helper functions for converting ADC readings to physical units
def ADCtoAccel(adc):
    '''
    Converts ADC readings from accelerometer to m/s^2
    Input:  adc - (int np.array shape (3, N)) ADC reading
    Output: acc - (float np.array shape (3, N)) acceleration in m/s^2
    '''
    bias        = np.array([0, 0, 0]).reshape(3,1)       # (mV)
    sensitivity = np.array([3023.8, 3023.8, 3023.8]).reshape(3,1) # (mV/grav)
    return (adc.astype(np.float64) - bias) * 3300 / (1023 * sensitivity) * 9.81

def ADCtoGyro(adc):
    '''
    Converts ADC readings from gyroscope to rad/s
    Input:  adc - (int np.array shape (3, N)) ADC reading
    Output: gyr - (float np.array shape (3, N)) angular velocity in rad/s
    '''
    bias        = np.array([0, 0, 0]).reshape(3,1)       # (mV)
    sensitivity = np.array([250, 250, 250]).reshape(3,1) # (mV/(rad/sec))
    return (adc.astype(np.float64) - bias) * 3300 / (1023 * sensitivity)

def VicontoRPY(vicon):
    '''
    COPILOT WRITTEN - CHECK THIS
    Converts Vicon rotation matrices to roll, pitch, yaw
    Input:  vicon - (float np.array shape (3, 3, N)) rotation matrices
    Output: roll  - (float np.array shape (N,)) roll angles in radians
    Output: pitch - (float np.array shape (N,)) pitch angles in radians
    Output: yaw   - (float np.array shape (N,)) yaw angles in radians
    '''
    roll  = np.arctan2(vicon[2,1,:], vicon[2,2,:])
    pitch = np.arctan2(-vicon[2,0,:], np.sqrt(vicon[2,1,:]**2 + vicon[2,2,:]**2))
    yaw   = np.arctan2(vicon[1,0,:], vicon[0,0,:])
    return roll, pitch, yaw

def plotStuff(accel, vicon_rpy=None):
    plt.figure(1)
    plt.plot(accel[0,:])
    plt.plot(accel[1,:])
    plt.plot(accel[2,:])
    plt.legend(['x', 'y', 'z'])
    plt.title('Accelerometer Data')
    plt.xlabel('Time (s)')
    plt.ylabel('Acceleration (m/s^2)')

    plt.figure(2)
    plt.plot(np.linalg.norm(accel, axis=0))
    plt.title('Norm of Accelerometer Data')
    plt.legend('magnitude of acceleration')
    plt.xlabel('Time (s)')
    plt.ylabel('Acceleration (m/s^2)')
    plt.show()

''' CAUTION MOVING FORWARD:
(1) The orientation of the IMU need not be the same as the orientation of the Vicon
coordinate frame. Plot all quantities in the arrays accel, gyro and vicon rotation
matrices to make sure you get this right. Do not proceed to implementing the filter if
you are not convinced your solution for this part is correct.

(2) The acceleration ax and ay is flipped in sign due to device design. A positive
acceleration in body-frame will result in a negative number reported by the IMU.
See the IMU manual for more insight.
'''


def estimate_rot(data_num=1):

    accel, gyro, T = load_imu_data(data_num)
    vicon = load_vicon_data(data_num)

    # Convert ADC readings to physical units
    accel = ADCtoAccel(accel)
    gyro  = ADCtoGyro(gyro)

    plotStuff(accel)

    # roll, pitch, yaw are numpy arrays of length T
    # return roll,pitch,yaw
    return 0

#### TESTING ####
if __name__ == '__main__':
    _ = estimate_rot(1)