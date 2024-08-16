import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy as sp
import sqlalchemy as sql
import datetime
from scipy.integrate import quad
from scipy.integrate import cumulative_trapezoid
from scipy.linalg import inv
# datetime does not matter in reading form csv, the plot function will handle it
# genfromtxt is outdated too, use pandas instead
#nrow is specified to read only 50 rows
# need to figure out  how to allow user to select which table they want from the sqlite database to load in




def kalman_filter_position_time(z, t, x0, P0, Q, R):
    """
    Implements the Kalman filter algorithm for position and time.
    
    z: measurements (position)
    t: time values
    x0: initial state estimate [position, time]
    P0: initial error covariance
    Q: process noise covariance
    R: measurement noise covariance
    """
    n = len(z)
    x = np.zeros((n, 2))
    P = np.zeros((n, 2, 2))
    
    x[0] = x0
    P[0] = P0
    
    for k in range(1, n):
        dt = t[k] - t[k-1]
        
        # State transition matrix
        F = np.array([[1, 0],
                      [0, 1]])
        
        # Measurement matrix
        H = np.array([[1, 0]])
        
        # Predict
        x_pred = F @ x[k-1]
        x_pred[1] = t[k]  # Update time directly
        P_pred = F @ P[k-1] @ F.T + Q
        
        # Update
        y = z[k] - H @ x_pred
        S = H @ P_pred @ H.T + R
        K = P_pred @ H.T @ inv(S)
        x[k] = x_pred + K @ y
        P[k] = (np.eye(2) - K @ H) @ P_pred
    
    return x


df = pd.read_csv('Data.csv',nrows = 500)
df["datetime"] = pd.to_datetime(df["datetime"])

dfcopy = df.copy()

num_rows = df.shape[0]

def timeOption(type, dataset, startTime=None):
    
    
    if startTime is None:
        startTime = dataset.iloc[0, 0]  # Use the first datetime value in the dataset as default
    else:
        date_format ='%Y-%m-%d %H:%M:%S'
        date_obj = datetime.datetime.strptime(startTime, date_format)

    index_vals =df.index[df['datetime'] >= startTime].tolist()
    index_lower= index_vals[0]
    index_upper= index_vals[-1]    
    # Create a new column to store the converted values
    dataset[type + '_converted'] = 0.0  # Initialize with float values
    
    for i in range(num_rows):
        if type == 'seconds':
            dataset.at[i, type + '_converted'] = float((dataset.iloc[i, 0] - date_obj).total_seconds())
        elif type == 'minutes':
            dataset.at[i, type + '_converted'] = float((dataset.iloc[i, 0] - date_obj).total_seconds() / 60)
        elif type == 'hours':
            dataset.at[i, type + '_converted'] = float((dataset.iloc[i, 0] - date_obj).total_seconds() / 3600)
        elif type == 'days':
            dataset.at[i, type + '_converted'] = float((dataset.iloc[i, 0] - date_obj).total_seconds() / 86400)
        else:
            print('Invalid time option')

    return (dataset,type + '_converted',index_lower,index_upper)




dfcopy,option,indexStart,indexEnd =timeOption('hours',dfcopy,'2015-01-01 6:00:00')

dfilter= dfcopy.loc[:, dfcopy.columns != 'machineID']

rotate_data = dfilter['rotate'].values
time_data = dfilter['hours_converted'].values


Q = np.eye(2) * 0.01  # process noise covariance
R = np.array([[0.1]])  # measurement noise covariance
x0 = np.array([rotate_data[0], time_data[0]])  # initial state estimate [position, time]
P0 = np.eye(2) * 1000  # initial error covariance

# Apply Kalman filter
filtered_state = kalman_filter_position_time(rotate_data, time_data, x0, P0, Q, R)



# Extract filtered position and time
filtered_position = filtered_state[:, 0]
filtered_time = filtered_state[:, 1]

# Calculate the gradient (derivative) of the filtered position
gradient = np.gradient(filtered_position, filtered_time)


# Calculate the integral of the filtered position
integral = cumulative_trapezoid(filtered_position, filtered_time, initial=0)
print(integral)

fig, axs = plt.subplots(4, 1, figsize=(15, 20), sharex=True)
# Plot original and filtered data
axs[0].plot(time_data, rotate_data, 'b.', label='Original Data', alpha=0.5)
axs[0].plot(filtered_time, filtered_position, 'r-', label='Kalman Filter Estimate')
axs[0].set_ylabel('Rotate')
axs[0].set_title('Kalman Filter Applied to Rotate Data (Position and Time)')
axs[0].legend()
axs[0].grid(True)

# Plot Kalman filter result
axs[1].plot(filtered_time, filtered_position, 'g-', label='Kalman Filter Estimate')
axs[1].set_ylabel('Filtered Rotate')
axs[1].set_title('Kalman Filter Estimate')
axs[1].legend()
axs[1].grid(True)

# Plot gradient of Kalman filter result
axs[2].plot(filtered_time, gradient, 'm-', label='Gradient of Kalman Filter Estimate')
axs[2].set_ylabel('Gradient')
axs[2].set_title('Gradient of Kalman Filter Estimate')
axs[2].legend()
axs[2].grid(True)

# Plot integral of Kalman filter result
axs[3].plot(filtered_time, integral, 'c-', label='Integral of Kalman Filter Estimate')
axs[3].set_xlabel('Time (hours)')
axs[3].set_ylabel('Integral')
axs[3].set_title('Integral of Kalman Filter Estimate')
axs[3].legend()
axs[3].grid(True)

plt.tight_layout()
plt.show()

# Calculate and print the root mean square error (RMSE)
rmse = np.sqrt(np.mean((rotate_data - filtered_position)**2))
print(f"Root Mean Square Error: {rmse}")
