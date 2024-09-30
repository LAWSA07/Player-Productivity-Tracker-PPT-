import cv2
import os
import torch
import numpy as np

class AutomaticObjectTracking:
    def __init__(self, input_file, tracker_index=None):
        # Available trackers in OpenCV
        self.tracker_list = ['BOOSTING', 'MIL', 'KCF', 'TLD', 'MEDIANFLOW', 'GOTURN', 'MOSSE', 'CSRT']
        self.tracker_type = self.tracker_list[tracker_index] if tracker_index is not None else 'CSRT'
        self.input_file = input_file
        self.tracker = self.initialize_tracker(self.tracker_type)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.load_yolo_model()

    def load_yolo_model(self):
        # Load YOLOv5 model
        print("Loading YOLOv5 model...")
        model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
        model.to(self.device)
        return model

    def initialize_tracker(self, tracker_type):
        # Initialize the OpenCV tracker based on type
        tracker_types = {
            'BOOSTING': cv2.legacy.TrackerBoosting_create,
            'MIL': cv2.legacy.TrackerMIL_create,
            'KCF': cv2.legacy.TrackerKCF_create,
            'TLD': cv2.legacy.TrackerTLD_create,
            'MEDIANFLOW': cv2.legacy.TrackerMedianFlow_create,
            'MOSSE': cv2.legacy.TrackerMOSSE_create,
            'CSRT': cv2.legacy.TrackerCSRT_create
        }

        create_tracker = tracker_types.get(tracker_type)
        if create_tracker is None:
            print(f"Tracker {tracker_type} not found. Using CSRT tracker.")
            create_tracker = cv2.legacy.TrackerCSRT_create

        return create_tracker()

    def get_bbox_from_yolov5(self, image):
        # Use YOLOv5 to get bounding box
        img_size = 640
        image_resized = cv2.resize(image, (img_size, img_size))
        results = self.model(image_resized)
        predictions = results.pred[0]

        if len(predictions) > 0:
            # Filter for person class (class index 0 in COCO dataset)
            person_detections = predictions[predictions[:, 5] == 0]
            if len(person_detections) > 0:
                best_detection = person_detections[0]
                x1, y1, x2, y2, conf, cls = best_detection.cpu().numpy()

                # Ensure the confidence threshold is high enough
                if conf > 0.5:
                    print(f"Detected person: conf={conf:.2f}, bbox=({x1},{y1},{x2},{y2})")

                    # Scale bbox to original image size
                    h, w = image.shape[:2]
                    bbox = (int(x1 * w / img_size), int(y1 * h / img_size),
                            int((x2 - x1) * w / img_size), int((y2 - y1) * h / img_size))
                    return bbox

        print("No person detected")
        return None

    def automatic_tracking_main(self):
        vid_frame = cv2.VideoCapture(self.input_file)
        if not vid_frame.isOpened():
            print(f'Error: Unable to open video file at {self.input_file}')
            return

        # Read the first frame
        ok, image = vid_frame.read()
        if not ok:
            print('Error: Unable to read video')
            return

        # Get initial bounding box using YOLOv5
        bbox = self.get_bbox_from_yolov5(image)
        if bbox is None:
            print("Failed to detect a person in the first frame. Exiting.")
            return

        print(f"Initializing tracker with bbox: {bbox}")
        ok = self.tracker.init(image, bbox)

        image_list = []
        tracking_data = []
        frame_count = 0
        while True:
            ok, image = vid_frame.read()
            if not ok:
                break

            # Measure FPS
            timer = cv2.getTickCount()
            ok, bbox = self.tracker.update(image)
            fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)

            # Draw the bounding box or show failure
            if ok:
                p1 = (int(bbox[0]), int(bbox[1]))
                p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
                cv2.rectangle(image, p1, p2, (255, 0, 0), 2, 1)
            else:
                cv2.putText(image, "Tracking failure detected", (100, 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)

            # Display tracker type and FPS on the image
            cv2.putText(image, f"Tracker: {self.tracker_type}", (100, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2)
            cv2.putText(image, f"FPS: {int(fps)}", (100, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2)

            # Show the tracking result
            cv2.imshow("Tracking", image)
            image_list.append(image)
            tracking_data.append(bbox)

            # Re-detect every 30 frames to update the bounding box
            frame_count += 1
            if frame_count % 30 == 0:
                bbox = self.get_bbox_from_yolov5(image)
                if bbox is not None:
                    ok = self.tracker.init(image, bbox)
                    print(f"Re-initialized tracker at frame {frame_count} with bbox: {bbox}")

            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        vid_frame.release()
        cv2.destroyAllWindows()
        self.save_tracked_video(image_list)

        return tracking_data

    def save_tracked_video(self, image_list):
        # Save the tracked frames to a video file
        if not image_list:
            print("No frames to save")
            return

        output_folder = 'data/output'
        if not os.path.exists(output_folder):
            os.makedirs(output_folder, 0o755)
        output_file_path = os.path.join(output_folder, "TrackedVideo.mp4")

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_file_path, fourcc, 30.0,
                              (image_list[0].shape[1], image_list[0].shape[0]))

        for frame in image_list:
            out.write(frame)

        out.release()
        print(f"Tracked video saved to {output_file_path}")

# Example usage:
# tracker = AutomaticObjectTracking(input_file='path/to/video.mp4', tracker_index=7)  # CSRT tracker
# tracking_data = tracker.automatic_tracking_main()
