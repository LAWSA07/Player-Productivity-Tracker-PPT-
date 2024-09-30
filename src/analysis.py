import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

class SprintAnalysis:
    def __init__(self, video_path, tracking_data):
        self.video_path = video_path
        self.tracking_data = tracking_data
        self.fps = self.get_video_fps()
        
    def get_video_fps(self):
        cap = cv2.VideoCapture(self.video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        return fps
    
    def calculate_speed(self):
        positions = np.array([(x + w/2, y + h/2) for x, y, w, h in self.tracking_data])
        if len(positions) < 2:
            print("Insufficient tracking data to calculate speed.")
            return np.array([])  # Return an empty array if not enough data

        distances = np.sqrt(np.sum(np.diff(positions, axis=0)**2, axis=1))
        speeds = distances * self.fps  # pixels per second
        return speeds
    
    def calculate_acceleration(self, speeds):
        if len(speeds) < 2:
            print("Insufficient speed data to calculate acceleration.")
            return np.array([])  # Return an empty array if not enough data
        
        return np.diff(speeds) * self.fps  # pixels per second^2
    
    def calculate_direction_changes(self):
        positions = np.array([(x + w/2, y + h/2) for x, y, w, h in self.tracking_data])
        if len(positions) < 2:
            print("Insufficient tracking data to calculate direction changes.")
            return np.array([])  # Return an empty array if not enough data

        vectors = np.diff(positions, axis=0)
        angles = np.arctan2(vectors[:, 1], vectors[:, 0])
        direction_changes = np.diff(angles)
        return np.abs(direction_changes)
    
    def smooth_data(self, data, window_length=5, polyorder=2):
        # Ensure the window_length does not exceed data length
        if len(data) < window_length:
            print(f"Data too short for smoothing with window length {window_length}.")
            return data  # Return the original data if it's too short
        return savgol_filter(data, window_length, polyorder)
    
    def analyze(self):
        speeds = self.calculate_speed()
        accelerations = self.calculate_acceleration(speeds)
        direction_changes = self.calculate_direction_changes()
        
        # Smooth the data
        smooth_speeds = self.smooth_data(speeds)
        smooth_accelerations = self.smooth_data(accelerations) if len(accelerations) > 0 else accelerations
        smooth_direction_changes = self.smooth_data(direction_changes) if len(direction_changes) > 0 else direction_changes
        
        return {
            'speeds': smooth_speeds,
            'accelerations': smooth_accelerations,
            'direction_changes': smooth_direction_changes
        }

    def visualize_results(self, results):
        # Only plot if there's enough data
        if len(results['speeds']) == 0:
            print("Not enough data to visualize results.")
            return
        
        time = np.arange(len(results['speeds'])) / self.fps
        
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 15))
        
        ax1.plot(time, results['speeds'], label='Speed')
        ax1.set_title('Speed over time')
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('Speed (pixels/s)')
        
        if len(results['accelerations']) > 0:
            time_accel = time[1:]  # Acceleration has one less element than speeds
            ax2.plot(time_accel, results['accelerations'], label='Acceleration')
            ax2.set_title('Acceleration over time')
            ax2.set_xlabel('Time (s)')
            ax2.set_ylabel('Acceleration (pixels/s^2)')
        else:
            ax2.set_title('Acceleration over time (Not enough data)')
        
        if len(results['direction_changes']) > 0:
            time_dir = time[1:]  # Direction changes have one less element than positions
            ax3.plot(time_dir, results['direction_changes'], label='Direction Change')
            ax3.set_title('Direction changes over time')
            ax3.set_xlabel('Time (s)')
            ax3.set_ylabel('Direction change (radians)')
        else:
            ax3.set_title('Direction changes over time (Not enough data)')
        
        plt.tight_layout()

        # Create the output directory if it doesn't exist
        output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'output')
        os.makedirs(output_dir, exist_ok=True)
        
        # Save the figure
        output_path = os.path.join(output_dir, 'sprint_analysis.png')
        plt.savefig(output_path)
        print(f"Analysis plot saved to: {output_path}")
        
        plt.show()

# Example usage:
# sprint_analysis = SprintAnalysis('path_to_video.mp4', tracking_data)
# results = sprint_analysis.analyze()
# sprint_analysis.visualize_results(results)
