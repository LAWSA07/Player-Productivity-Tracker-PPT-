import cv2

def select_player(frame):
    bbox = cv2.selectROI(frame, False)
    return bbox
