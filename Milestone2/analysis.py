
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# Define paths
DATA_DIR = 'Milestone2/lab9-dataset'
ACCELERATION_FILE = os.path.join(DATA_DIR, 'ACCELERATION.csv')
WALKING_FILE = os.path.join(DATA_DIR, 'WALKING.csv')
TURNING_FILE = os.path.join(DATA_DIR, 'TURNING.csv')
WALKING_AND_TURNING_FILE = os.path.join(DATA_DIR, 'WALKING_AND_TURNING.csv')

def part1_sensor_errors():
    print("--- Part 1: Understanding Sensor Data Errors ---")
    if not os.path.exists(ACCELERATION_FILE):
        print(f"File not found: {ACCELERATION_FILE}")
        return

    # Load data
    df = pd.read_csv(ACCELERATION_FILE)
    
    # Extract columns
    # Assuming columns are: timestamp, acceleration, noisyacceleration
    # Check column names first or assume based on spec
    # Spec says: "The first row of the line indicates these field names."
    
    time = df['timestamp'].values
    acc_real = df['acceleration'].values
    acc_noisy = df['noisyacceleration'].values
    
    # Calculate dt (assuming constant 0.1s or calculate from time)
    # Spec says 0.1s sampling rate. Let's verify or just use 0.1
    dt = 0.1
    
    # Calculate Velocity (Integration of Acceleration)
    # v = v0 + a * dt
    vel_real = np.cumsum(acc_real) * dt
    vel_noisy = np.cumsum(acc_noisy) * dt
    
    # Calculate Distance (Integration of Velocity)
    # d = d0 + v * dt
    dist_real = np.cumsum(vel_real) * dt
    dist_noisy = np.cumsum(vel_noisy) * dt
    
    # Report final distances
    final_dist_real = dist_real[-1]
    final_dist_noisy = dist_noisy[-1]
    diff = abs(final_dist_real - final_dist_noisy)
    
    print(f"Final Distance (Real): {final_dist_real:.4f} m")
    print(f"Final Distance (Noisy): {final_dist_noisy:.4f} m")
    print(f"Difference: {diff:.4f} m")
    
    # Plotting
    plt.figure(figsize=(15, 10))
    
    # Plot 1: Acceleration
    plt.subplot(3, 1, 1)
    plt.plot(time, acc_real, label='Real Acceleration', color='blue')
    plt.plot(time, acc_noisy, label='Noisy Acceleration', color='red', alpha=0.7)
    plt.title('Acceleration vs Time')
    plt.ylabel('Acceleration (m/s^2)')
    plt.legend()
    plt.grid(True)
    
    # Plot 2: Velocity
    plt.subplot(3, 1, 2)
    plt.plot(time, vel_real, label='Real Velocity', color='blue')
    plt.plot(time, vel_noisy, label='Noisy Velocity', color='red', alpha=0.7)
    plt.title('Velocity vs Time')
    plt.ylabel('Velocity (m/s)')
    plt.legend()
    plt.grid(True)
    
    # Plot 3: Distance
    plt.subplot(3, 1, 3)
    plt.plot(time, dist_real, label='Real Distance', color='blue')
    plt.plot(time, dist_noisy, label='Noisy Distance', color='red', alpha=0.7)
    plt.title('Distance vs Time')
    plt.xlabel('Time (s)')
    plt.ylabel('Distance (m)')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('Milestone2/part1_plots.png')
    print("Plots saved to Milestone2/part1_plots.png")
    # plt.show() # Comment out for non-interactive run

from scipy.signal import find_peaks

def part2_step_detection():
    print("\n--- Part 2: Step Detection ---")
    if not os.path.exists(WALKING_FILE):
        print(f"File not found: {WALKING_FILE}")
        return

    # Load data
    df = pd.read_csv(WALKING_FILE)
    time = df['timestamp'].values
    # Convert timestamp to seconds (start from 0)
    time = (time - time[0]) / 1e9 
    
    acc_x = df['accel_x'].values
    acc_y = df['accel_y'].values
    acc_z = df['accel_z'].values
    
    # Calculate Magnitude
    acc_mag = np.sqrt(acc_x**2 + acc_y**2 + acc_z**2)
    
    # Smooth data (Moving Average)
    window_size = 20 # Adjust as needed
    acc_smooth = pd.Series(acc_mag).rolling(window=window_size, center=True).mean().values
    
    # Detect Steps (Peaks)
    # Parameters need tuning based on data
    # Gravity is ~9.8. Steps usually go above 10-11.
    # Min distance: 0.3-0.5s. Sampling is ~5ms (200Hz). 0.4s = 80 samples.
    peaks, _ = find_peaks(acc_smooth, height=10.5, distance=60)
    
    step_count = len(peaks)
    print(f"Step Count: {step_count}")
    
    # Plotting
    plt.figure(figsize=(15, 6))
    plt.plot(time, acc_mag, label='Raw Acceleration Magnitude', alpha=0.3, color='gray')
    plt.plot(time, acc_smooth, label='Smoothed Magnitude', color='blue', linewidth=2)
    plt.plot(time[peaks], acc_smooth[peaks], "x", label='Detected Steps', color='red', markersize=10)
    plt.title(f'Step Detection (Count: {step_count})')
    plt.xlabel('Time (s)')
    plt.ylabel('Acceleration (m/s^2)')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('Milestone2/part2_plots.png')
    print("Plots saved to Milestone2/part2_plots.png")

def part3_direction_detection():
    print("\n--- Part 3: Direction Detection ---")
    if not os.path.exists(TURNING_FILE):
        print(f"File not found: {TURNING_FILE}")
        return

    # Load data
    print(f"Loading {TURNING_FILE}...")
    try:
        df = pd.read_csv(TURNING_FILE, on_bad_lines='skip', engine='python')
        print(f"Loaded {len(df)} rows.")
    except Exception as e:
        print(f"Error loading TURNING_FILE: {e}")
        return
    time = df['timestamp'].values
    time = (time - time[0]) / 1e9
    
    gyro_z = df['gyro_z'].values
    
    # Calculate dt
    dt = np.diff(time)
    dt = np.insert(dt, 0, 0) # Insert 0 at beginning to match length
    
    # Integrate Gyro Z to get Angle (in radians)
    # angle = cumsum(gyro * dt)
    angle_rad = np.cumsum(gyro_z * dt)
    angle_deg = np.degrees(angle_rad)
    
    # Detect Turns
    # Turns are characterized by peaks in angular velocity (gyro_z)
    # We expect 4 clockwise and 4 counter-clockwise turns.
    # Clockwise might be negative or positive depending on axis.
    # Let's look at absolute gyro_z to find all turns.
    
    # Smooth gyro data for peak detection
    gyro_smooth = pd.Series(gyro_z).rolling(window=50, center=True).mean().fillna(0).values
    
    # Detect peaks in absolute angular velocity
    # Threshold: Turns are usually fast. > 0.5 rad/s?
    # Distance: Turns are separated.
    peaks, _ = find_peaks(np.abs(gyro_smooth), height=0.5, distance=100)
    
    print(f"Detected {len(peaks)} turns.")
    
    # Calculate turn angles
    # For each peak, integrate gyro_z over a window around the peak
    turn_angles = []
    for p in peaks:
        # Define window around peak (e.g., +/- 100 samples or until zero crossing)
        # Simple approach: fixed window
        window = 150
        start = max(0, p - window)
        end = min(len(time), p + window)
        
        # Integrate gyro_z in this window
        # Better: integrate from start of turn to end of turn.
        # But fixed window is a good approximation if peaks are isolated.
        turn_angle_rad = np.sum(gyro_z[start:end] * dt[start:end])
        turn_angle_deg = np.degrees(turn_angle_rad)
        turn_angles.append(turn_angle_deg)
        print(f"Turn at {time[p]:.2f}s: {turn_angle_deg:.2f} degrees")
        
    # Plotting
    plt.figure(figsize=(15, 8))
    
    plt.subplot(2, 1, 1)
    plt.plot(time, gyro_z, label='Raw Gyro Z', alpha=0.5, color='gray')
    plt.plot(time, gyro_smooth, label='Smoothed Gyro Z', color='blue')
    plt.plot(time[peaks], gyro_smooth[peaks], "x", label='Detected Turns', color='red', markersize=10)
    plt.title('Gyroscope Z (Angular Velocity)')
    plt.ylabel('rad/s')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(2, 1, 2)
    plt.plot(time, angle_deg, label='Integrated Angle', color='green')
    plt.title('Integrated Angle (Direction)')
    plt.xlabel('Time (s)')
    plt.ylabel('Degrees')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('Milestone2/part3_plots.png')
    print("Plots saved to Milestone2/part3_plots.png")

def part4_trajectory():
    print("\n--- Part 4: Trajectory Plotting ---")
    if not os.path.exists(WALKING_AND_TURNING_FILE):
        print(f"File not found: {WALKING_AND_TURNING_FILE}")
        return

    # Load data
    try:
        df = pd.read_csv(WALKING_AND_TURNING_FILE, on_bad_lines='skip', engine='python')
    except Exception as e:
        print(f"Error reading file: {e}")
        return

    time = df['timestamp'].values
    time = (time - time[0]) / 1e9
    
    acc_x = df['accel_x'].values
    acc_y = df['accel_y'].values
    acc_z = df['accel_z'].values
    gyro_z = df['gyro_z'].values
    
    # 1. Step Detection
    acc_mag = np.sqrt(acc_x**2 + acc_y**2 + acc_z**2)
    window_size = 20
    acc_smooth = pd.Series(acc_mag).rolling(window=window_size, center=True).mean().values
    # Detect steps
    step_peaks, _ = find_peaks(acc_smooth, height=10.5, distance=60)
    print(f"Detected {len(step_peaks)} steps.")
    
    # 2. Orientation
    dt = np.diff(time)
    dt = np.insert(dt, 0, 0)
    angle_rad = np.cumsum(gyro_z * dt)
    # Initial orientation: North (90 degrees in standard cartesian, or 0 if we plot Y as North)
    # Let's assume 0 is East, 90 is North.
    # Initial orientation is North -> start at pi/2
    initial_angle = np.pi / 2
    angle_rad += initial_angle
    
    # 3. Reconstruct Path
    x = [0]
    y = [0]
    step_length = 0.75 # meters (estimated)
    
    current_x = 0
    current_y = 0
    
    for step_idx in step_peaks:
        # Get orientation at this step
        theta = angle_rad[step_idx]
        
        # Update position
        dx = step_length * np.cos(theta)
        dy = step_length * np.sin(theta)
        
        current_x += dx
        current_y += dy
        
        x.append(current_x)
        y.append(current_y)
        
    # Plotting
    plt.figure(figsize=(10, 10))
    plt.plot(x, y, marker='o', markersize=4, linestyle='-', label='Trajectory')
    plt.plot(x[0], y[0], 'go', markersize=10, label='Start')
    plt.plot(x[-1], y[-1], 'rx', markersize=10, label='End')
    
    plt.title(f'Reconstructed Trajectory ({len(step_peaks)} steps)')
    plt.xlabel('X Position (m)')
    plt.ylabel('Y Position (m)')
    plt.axis('equal')
    plt.grid(True)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('Milestone2/part4_trajectory.png')
    print("Plots saved to Milestone2/part4_trajectory.png")

if __name__ == "__main__":
    part1_sensor_errors()
    part2_step_detection()
    part3_direction_detection()
    part4_trajectory()
