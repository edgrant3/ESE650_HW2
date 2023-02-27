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
    ts = imu['ts'][0] - imu['ts'][0][0] # (5645,) array of timestamps
    return accel, gyro, T, ts

def load_vicon_data(data_num):
    # Load "ground truth" VICON data (COMMENT OUT FOR AUTOGRADER SUBMISSION)
    # dict with keys 'rots' and 'ts' := 
    # (3, 3, 5561) array of rotations | (1x5561) array of timestamps
    vicon = io.loadmat('vicon/viconRot' + str(data_num) + '.mat')
    tv = vicon['ts'][0] - vicon['ts'][0][0] # (5645,) array of timestamps
    return vicon, tv

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

def preprocess_dataset(data_num):
    # Load IMU data
    accel, gyro, nT, T_sensor = load_imu_data(data_num)
    vicon, T_vicon = load_vicon_data(data_num)
    calibrationPrint(accel, gyro, 'before transform')

    # Convert ADC readings to physical units
    accel = ADCtoAccel(accel)
    gyro  = ADCtoGyro(gyro)
    calibrationPrint(accel, gyro, 'after transform')

    # Convert vicon rotation matrices to roll, pitch, yaw
    roll, pitch, yaw = VicontoRPY(vicon['rots'])

    plotStuff(accel, T_sensor.ravel(), roll, pitch, yaw, T_vicon.ravel())

    return accel, gyro, T_sensor, roll, pitch, yaw, T_vicon

def plotStuff(accel, ts, roll, pitch, yaw, tv):
    plt.figure(1)
    plt.plot(ts, accel[0,:])
    plt.plot(ts, accel[1,:])
    plt.plot(ts, accel[2,:])
    plt.legend(['x', 'y', 'z'])
    plt.title('Accelerometer Data')
    plt.xlabel('Time (s)')
    plt.ylabel('Acceleration (m/s^2)')

    plt.figure(2)
    plt.plot(ts, np.ones(accel.shape[1]) * 9.81, 'k--')
    plt.plot(ts, np.linalg.norm(accel, axis=0), alpha=0.5)
    plt.title('Norm of Accelerometer Data')
    plt.legend(['magnitude of acceleration', '|g| (9.81 m/s^2)'])
    plt.xlabel('Time (s)')
    plt.ylabel('Acceleration (m/s^2)')
    
    plt.figure(3)
    plt.plot(tv, roll, alpha=0.75)
    plt.plot(tv, pitch, alpha=0.75)
    plt.plot(tv, yaw, alpha=0.75)
    plt.plot(tv, np.ones_like(tv) * np.pi / 2, 'k--', alpha=0.5)
    plt.plot(tv, np.ones_like(tv) * (-np.pi) / 2, 'k--', alpha=0.5)
    plt.plot(tv, np.ones_like(tv) * np.pi, 'k--', alpha=0.5)
    plt.plot(tv, np.ones_like(tv) * (-np.pi), 'k--', alpha=0.5)
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

class State:
    def __init__(self, quat_vec, omegas):
        self.quat = Quaternion(quat_vec[0].astype(np.float64), quat_vec[1:])
        self.w = omegas
        self.quat_state_vec = self.with_quat()
        # self.axis_angle_state_vec = self.with_axis_angle()

    def with_quat(self):
        # gives (7,1) state vector with first four elements as quaternion elements
        return np.hstack((self.q.q, self.w)).reshape(self.quat.q.size+self.w.size, 1)

    def with_axis_angle(self):
        # replaces 4-element quaternion portion of state with its axis-angle represntation.
        # This reduces the state dimensions from (7,1) to (6,1)
        return np.hstack((self.q.axis_angle(), self.w)).reshape(self.quat.q.size+self.w.size, 1)
        

def generate_sigma_points(mean, cov):
    # Mean is State object (where mean.q.shape = (7,1)), and cov is (6,6) numpy array
    # n is no. of covariance columns
    n = cov.shape[1]

    # offset is (n,2n) array where rows of sqrt(n*cov) become 
    # columns added to mean to create sigma points
    offset = (np.sqrt(n) * linalg.sqrtm(cov)).T
    offset = np.hstack((offset, -offset))

    # initialize sigma points of shape (7,2n)
    # and merely add the angular velocity (last 3) copmponents to the offset
    sig_pts = np.zeros((mean.quat_state_vec.shape[0], 2*n))
    sig_pts[-3:, :] += offset[-3:, :]

    # must convert first 3 elements of offset term to 4-element quaternion
    # then "add" them via quaternion multiplication
    for i in range(sig_pts.shape[1]):
        offset_quat = Quaternion().from_axis_angle(offset[0:3, i])
        combo_quat  = offset_quat.__mul__(mean)
        sig_pts[0:4, i] = combo_quat.q

    return sig_pts


def compute_GD_update(sig_pts, prev_state, threshold = 0.1):

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
    #can use np.cov
    new_cov[:3, :3] = (E - e_bar) @ (E - e_bar).T / sig_pts.shape[1] #(3, 2n) @ (2n, 3) = (3, 3)
    new_cov[3:, 3:] =(sig_pts[4:, :] - new_mean[4:]) @ (sig_pts[4:, :] - new_mean[4:]).T / sig_pts.shape[1] # + R

    return new_mean, new_cov


def propagate_dynamics(sp, dt, R, use_noise=False):
    
    sp_propagated = np.zeros(sp.shape)

    if use_noise:
        # add noise to each sigma point along with dynamics
        rng = np.random.default_rng(1998)
        noise = np.zeros((sp.shape[0]-1, 1))       

        for i in range(sp.shape[1]):
            
            for n_idx in range(sp.shape[0]-1):
                noise[n_idx] = rng.normal(0, R[n_idx,n_idx], ())

            q_delta = Quaternion().from_axis_angle(sp[-3:, i] * dt)
            q_noise = Quaternion().from_axis_angle(noise[0:3, 0])
            q_sp = Quaternion(np.float64(sp[0, i]), sp[1:4, i].ravel())

            q_comb = q_sp.__mul__(q_noise.__mul__(q_delta))

            sp_propagated[0:4,i] = q_comb.q
            sp_propagated[4:,i] = sp[4:,i] + noise[3:,0]

        return sp_propagated

    else:
        # just propagate sigma points through dynamics
        for i in range(sp.shape[1]):

            q_delta = Quaternion().from_axis_angle(sp[-3:, i] * dt)
            q_sp = Quaternion(np.float64(sp[0, i]), sp[1:4, i].ravel())

            q_comb = q_sp.__mul__(q_delta)

            sp_propagated[0:4,i] = q_comb.q
            sp_propagated[4:,i] = sp[4:,i]

        return sp_propagated
    
def propagate_measurement(sp, Q, use_noise=False):
    return 0

def estimate_rot(data_num=1):

    accel, gyro, T_sensor, roll_gt, pitch_gt, yaw_gt, T_vicon = preprocess_dataset(data_num)

    ### (1) Initialize Parameters
    # init state
    state_0 = State(np.array([1, 0, 0, 0]), np.array([0, 0, 0]))
    # init covariance matrix
    cov_0 = np.eye(6, 6)
    # init process noise covariance matrix
    R = np.diag([0.05, 0.05, 0.05, 0.10, 0.10, 0.10]])
    # init measurement noise covariance matrix
    Q = np.diag([0.05, 0.05, 0.05, 0.10, 0.10, 0.10]])

    means = [state_0]
    mean_k_k = state_0.quat_state_vec
    cov_k_k = cov_0
    
    for t in range(T_sensor.size - 1):
        ### (2) Add Noise Component to Covariance
        dt = T_sensor[t+1] - T_sensor[t]
        cov_k_k += R * dt

        ### (3) Generate Sigma Points
        sp = generate_sigma_points(mean_k_k, cov_k_k)

        ### (4) Propagate Sigma Points Thru Dynamics
        sp_propagated = propagate_dynamics(sp, dt, R, means[t], use_noise=False)

        ### (5) Compute Mean and Covariance of Propagated Sigma Points
        mean_k1_k, cov_k1_k = compute_GD_update(sp_propagated, means[t])

        ### (6) Compute Sigma Points with Updated Mean and Covariance
        sp = generate_sigma_points(mean_k1_k, cov_k1_k)

        ### (7) Propagate Sigma Points Thru Measurement Model
        sp_propagated = propagate_measurement(sp, dt, R, means[t], use_noise=False)



    # roll, pitch, yaw are numpy arrays of length T
    # return roll,pitch,yaw
    return 0

#### TESTING ####
if __name__ == '__main__':
    _ = estimate_rot(1)

    # test_mean = np.array([0, 1, 2, 3, 4, 5, 6]).reshape(7,1) + 3
    # test_cov  = np.eye(6,6)
    # print(test_mean)
    # print(test_cov)

    # test_sig_pts = generate_sigma_points(test_mean, test_cov)
    # print(test_sig_pts.shape)