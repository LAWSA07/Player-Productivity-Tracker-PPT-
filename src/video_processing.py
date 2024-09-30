import cv2

def read_video(input_file):
    vid_frame = cv2.VideoCapture(input_file)
    if not vid_frame.isOpened():
        print('Error in opening the video file')
        return None
    return vid_frame
