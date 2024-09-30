import sys
import os

from src.tracking import AutomaticObjectTracking
from src.analysis import SprintAnalysis

# Add the src directory to the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(current_dir, 'src')
sys.path.insert(0, src_dir)

if __name__ == '__main__':
    input_path = 'data/input/lin.mp4'  # Update with your actual video path
    objTrack = AutomaticObjectTracking(input_path, tracker_index=7)  # Using CSRT tracker (index 7)
    tracking_data = objTrack.automatic_tracking_main()
    
    analysis = SprintAnalysis(input_path, tracking_data)
    results = analysis.analyze()
    analysis.visualize_results(results)




