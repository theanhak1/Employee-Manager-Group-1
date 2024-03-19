import os
import sys
import threading
import cv2
from PIL import Image
from ultralytics import YOLO

# Configuration Constants
VIDEO_FILE_1 = "greet_detection/videos/istockphoto-1474996250-640_adpp_is.mp4"
MODEL_FILE = 'yolov8n.pt'
CLASSES_IDX = 0
DEVICE_ID = "0"


def initialize_model(model_file):
    """Initialize YOLO model."""
    return YOLO(model_file)


def initialize_video_capture(filename, width=None, height=None):
    """Initialize video capture."""    
    video = cv2.VideoCapture(filename)
    
    # Get the original width and height
    original_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    if width is not None and height is not None:
        # User specified both width and height, set them directly
        video.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        video.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    elif width is not None:
        # User specified only width, calculate height to maintain aspect ratio
        height = int((width / original_width) * original_height)
        video.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        video.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    elif height is not None:
        # User specified only height, calculate width to maintain aspect ratio
        width = int((height / original_height) * original_width)
        video.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        video.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    return video, width, height


def process_video(video, model, file_index, visualize: bool=False, verbose: bool=True, confidence: float=0.5):
    """Process individual video frame."""
    ret, frame = video.read()
    full_results = []
    if ret:
        results = model.track(frame,
                              persist=True,
                              classes=CLASSES_IDX, 
                              device=DEVICE_ID,
                              stream=True,
                              conf=confidence,
                              half=True,
                              max_det=10,
                              vid_stride=2,
                              tracker="bytetrack.yaml",
                              verbose=verbose)
        for idx, result in enumerate(results):
            full_results.append(result) # Convert from generator to list for easier access
            if visualize:
                if idx == 0: first_result = result
                res_plotted = first_result.plot()
                cv2.imshow(f"Tracking_Stream_{file_index}", res_plotted)
                key = cv2.waitKey(1)

                if key == ord('q'):
                    return False  # Break the loop if 'q' is pressed

    return ret, frame, full_results


def release_video(video):
    """Release video sources."""
    video.release()

def run_tracker_in_thread(filename, model, file_index, return_cropped_boxes=False, visualize=False):
    """Run video processing concurrently with YOLOv8 model using threading."""
    video = initialize_video_capture(filename)
    
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    
    while True: 
        ret, frame, results = process_video(video, model, file_index, visualize=visualize)
        if not ret:
            break
        if return_cropped_boxes:
            if visualize:
                ax.clear()  # Clear the previous frame
                for result in results:
                    pil_images = process_yolo_boxes(result, frame) 
                    for pil_image in pil_images:
                        ax.imshow(pil_image)
                        ax.axis('off')  # Optional: Turn off axis
                        plt.pause(0.0001)  # Pause for a short duration to show the frame
            
    release_video(video)

    return results, pil_images

def process_yolo_boxes(result, frame):
    """Extract object from YOLO bounding boxes."""
    pil_images = []
    for box in result.boxes.xyxy:
        x, y, x_max, y_max = box
        cropped_image = frame[int(y):int(y_max), int(x):int(x_max)]
        pil_image = Image.fromarray(cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB))
        pil_images.append(pil_image)
    
    return pil_images


def main():
    """Main function."""
    # Load the model
    yolo_model = initialize_model(MODEL_FILE)

    # # Create the tracker threads
    # tracker_thread1 = threading.Thread(target=run_tracker_in_thread, args=(VIDEO_FILE_1, yolo_model, 1, True, True), daemon=True)

    # # Start the tracker threads
    # tracker_thread1.start()

    # # Wait for the tracker threads to finish
    # tracker_thread1.join()
    run_tracker_in_thread(VIDEO_FILE_1, yolo_model, 1, True, True)

    # Clean up and close windows
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
