import numpy as np
from scipy import io
from scipy import linalg
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
    bias        = np.array([510.808, 500.994, 499]).reshape(3,1)       # (mV)
    sensitivity = np.array([340.5, 340.5, 342.25]).reshape(3,1) # (mV/grav)
    return (adc.astype(np.float64) - bias) * 3300 / (1023 * sensitivity) * 9.81

def ADCtoGyro(adc, convert_to_rad=True):
    '''
    Converts ADC readings from gyroscope to rad/s
    Input:  adc - (int np.array shape (3, N)) ADC reading
    Output: gyr - (float np.array shape (3, N)) angular velocity in rad/s
    z,x,y ordering!!!
    '''
    bias        = np.array([369.68, 373.568, 375.356]).reshape(3,1)       # (mV)
    sensitivity = np.array([200, 200, 200]).reshape(3,1) # (mV/(rad/sec))
    if convert_to_rad:
        return (adc.astype(np.float64) - bias) * 3300 / (1023 * sensitivity) 
    else:
        return (adc.astype(np.float64) - bias) * 3300 / (1023 * sensitivity) * 180 / np.pi 

def VicontoRPY(vicon_rots):
    '''
    COPILOT WRITTEN - CHECK THIS
    Converts Vicon rotation matrices to roll, pitch, yaw
    Input:  vicon - (float np.array shape (3, 3, N)) rotation matrices
    Output: roll  - (float np.array shape (N,)) roll angles in radians
    Output: pitch - (float np.array shape (N,)) pitch angles in radians
    Output: yaw   - (float np.array shape (N,)) yaw angles in radians
    '''
    N = vicon_rots.shape[2]
    roll  = np.zeros((N,))
    pitch = np.zeros((N,))
    yaw   = np.zeros((N,))

    for i in range(N):
        q = Quaternion()
        q.from_rotm(vicon_rots[:,:,i])
        roll[i], pitch[i], yaw[i] = q.euler_angles()
    return roll, pitch, yaw

def plotStuff(accel, ts, roll, pitch, yaw):
    plt.figure(1)
    plt.plot(accel[0,:])
    plt.plot(accel[1,:])
    plt.plot(accel[2,:])
    plt.legend(['x', 'y', 'z'])
    plt.title('Accelerometer Data')
    plt.xlabel('Time (s)')
    plt.ylabel('Acceleration (m/s^2)')

    plt.figure(2)
    plt.plot(np.ones(accel.shape[1]) * 9.81, 'k--')
    plt.plot(np.linalg.norm(accel, axis=0), alpha=0.5)
    plt.title('Norm of Accelerometer Data')
    plt.legend(['magnitude of acceleration', '|g| (9.81 m/s^2)'])
    plt.xlabel('Time (s)')
    plt.ylabel('Acceleration (m/s^2)')
    
    plt.figure(3)
    plt.plot(ts, roll, alpha=0.75)
    plt.plot(ts, pitch, alpha=0.75)
    plt.plot(ts, yaw, alpha=0.75)
    plt.plot(ts, np.ones_like(ts) * np.pi / 2, 'k--', alpha=0.5)
    plt.plot(ts, np.ones_like(ts) * (-np.pi) / 2, 'k--', alpha=0.5)
    plt.plot(ts, np.ones_like(ts) * np.pi, 'k--', alpha=0.5)
    plt.plot(ts, np.ones_like(ts) * (-np.pi), 'k--', alpha=0.5)
    plt.title('RPY of Vicon')
    plt.legend(['Roll', 'Pitch', 'Yaw'])
    plt.xlabel('Time (s)')
    plt.ylabel('Radians')
    plt.show()

def calibrationPrint(accel, gyro, str=''):
    print("Average of first 500 timesteps:")
    print(f'x accel avg {str}: {np.mean(accel[0,0:500])}')
    print(f'y accel avg {str}: {np.mean(accel[1,0:500])}')
    print(f'z accel avg {str}: {np.mean(accel[2,0:500])}')
    print(f'z gyro avg {str}: {np.mean(gyro[0,0:500])}')
    print(f'x gyro avg {str}: {np.mean(gyro[1,0:500])}')
    print(f'y gyro avg {str}: {np.mean(gyro[2,0:500])}\n')

''' CAUTION MOVING FORWARD:
(1) The orientation of the IMU need not be the same as the orientation of the Vicon
coordinate frame. Plot all quantities in the arrays accel, gyro and vicon rotation
matrices to make sure you get this right. Do not proceed to implementing the filter if
you are not convinced your solution for this part is correct.

(2) The acceleration ax and ay is flipped in sign due to device design. A positive
acceleration in body-frame will result in a negative number reported by the IMU.
See the IMU manual for more insight.
'''

def generate_sigma_points(mean, cov):
    # Mean is (7,1) array, and cov is (6,6)
    
    # n is no. of covariance columns
    n = cov.shape[1]

    # offset is (n,2n) array where rows of sqrt(n*cov) become 
    # columns added to mean to create sigma points
    offset = (np.sqrt(n) * linalg.sqrtm(cov)).T
    offset = np.hstack((offset, -offset))

    sig_pts = np.zeros((n+1, 2*n))
    sig_pts[-3:, :] += offset[-3:, :]

    # must convert first 3 elements of offset term to 4-element quaternion
    mean_quat = Quaternion(np.float64(mean[0]), mean[1:4].ravel())
    for i in range(sig_pts.shape[1]):
        offset_quat = Quaternion(0, offset[0:3, i])
        combo_quat  = offset_quat.__mul__(mean_quat)
        sig_pts[0:4, i] = combo_quat.q

    return sig_pts

def compute_GD_update(sig_pts, prev_state, R, threshold = 0.1):

    # Initialize mean quat to previous state's quaternion
    q_bar = Quaternion(np.float64(prev_state[0]), prev_state[1:4].ravel())
    # Initialize error matrix (contains axis-angle representation of error)
    E = np.ones(3, sig_pts.shape[1]) * np.inf
    mean_err = np.inf

    # Iterate until error is below threshold
    while mean_err > threshold:
        for i in range(sig_pts.shape[1]):
            # Convert sigma point to quaternion
            q_i = Quaternion(np.float64(sig_pts[0, i]), sig_pts[1:4, i].ravel())
            # Compute error quaternion
            q_err = q_i.__mul__(q_bar.inverse())
            # Convert error quaternion to axis-angle representation
            E[:, i] = q_err.axis_angle()

        e_bar = np.mean(E, axis=1)
        q_bar = q_bar.from_axis_angle(e_bar)
        mean_err = np.linalg.norm(e_bar)

    new_mean = np.zeros(7)
    new_mean[0:4] = q_bar.q
    new_mean[4:] = np.mean(sig_pts[4:, :], axis=1)

    new_cov = np.zeros((6,6))
    new_cov[:3, :3] = (E - e_bar) @ (E - e_bar).T / sig_pts.shape[1] #(3, 2n) @ (2n, 3) = (3, 3)
    new_cov[3:, 3:] = R + (sig_pts[4:, :] - new_mean[4:]) @ (sig_pts[4:, :] - new_mean[4:]).T / sig_pts.shape[1]

    return new_mean, new_cov
        




def estimate_rot(data_num=1):

    accel, gyro, T = load_imu_data(data_num)
    vicon = load_vicon_data(data_num)
    calibrationPrint(accel, gyro, 'before transform')

    # Convert ADC readings to physical units
    accel = ADCtoAccel(accel)
    gyro  = ADCtoGyro(gyro)
    calibrationPrint(accel, gyro, 'after transform')


    roll, pitch, yaw = VicontoRPY(vicon['rots'])

    plotStuff(accel, vicon['ts'].ravel(), roll, pitch, yaw)

    # roll, pitch, yaw are numpy arrays of length T
    # return roll,pitch,yaw
    return 0

#### TESTING ####
if __name__ == '__main__':
    # _ = estimate_rot(1)

    test_mean = np.array([0, 1, 2, 3, 4, 5, 6]).reshape(7,1) + 3
    test_cov  = np.eye(6,6)
    print(test_mean)
    print(test_cov)

    test_sig_pts = generate_sigma_points(test_mean, test_cov)
    print(test_sig_pts.shape)