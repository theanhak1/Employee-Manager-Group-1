import datetime
import sys
import time
import logging
import threading
import queue
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import cv2
sys.path.append('./')
from ultralytics.utils.plotting import Annotator
from face_verification import run_face_verification, create_embedding
from human_detection import process_video, process_yolo_boxes, initialize_model, initialize_video_capture, release_video

# Configuration Constants
VIDEO_FILE_1 = 0 # Path or int 
MODEL_FILE = 'yolov8n.pt'
DEVICE_ID = "0"

# Configure logging
logging.basicConfig(filename='identification_log.txt', level=logging.INFO, format='%(asctime)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

def human_detection_thread(video_file, model_file, results_queue, identify_queue, visualize=False, width=None, height=None, process_interval=5):
    """
    Thread function for human detection.

    Args:
        video_file (str): Path to the video file.
        model_file (str): Path to the model file.
        results_queue (Queue): Queue to store the processed video frames and results.
        identify_queue (Queue): Queue to retrieve identification results.
        visualize (bool, optional): Whether to visualize the detection results. Defaults to False.
        width (int, optional): Width of the video frame. Defaults to 512.
        height (int, optional): Height of the video frame. Defaults to None.
        process_interval (int, optional): Interval (in seconds) between processing frames. Defaults to 5.
    """
    model = initialize_model(model_file)
    video, _, _ = initialize_video_capture(video_file, width, height)
    track_id_name_map = {}
    start_time = time.time()
    while True:
        ret, frame, results = process_video(video, model, 0, visualize=False)
        if not ret:
            break
        track_ids = results[0].boxes.id.int().cpu().tolist() if results[0].boxes.id is not None else [] # Get track IDs
        current_time = time.time()
        if current_time - start_time > process_interval:
            results_queue.put((frame, results, track_ids))
            start_time = current_time
        # Retrieve identification results and plot
        if not identify_queue.empty():
            track_id_name_map = identify_queue.get()
        result_frame = plot_results(results, frame, track_ids, track_id_name_map)
        # Display the frame
        cv2.imshow("Frame", result_frame)
        cv2.waitKey(1)  # Display the frame for 1 millisecond

    release_video(video)
    results_queue.put(None) # Signal that processing is complete

def plot_results(results, frame, track_ids, track_id_name_map=None, colors=(0, 0, 255), line_thickness=3, font_scale=0.3):
    """
    Plot the results on the frame with bounding boxes and labels.

    Args:
        results (list): List of bounding box coordinates and class labels.
        frame (numpy.ndarray): The input frame/image.
        track_ids (list, optional): List of track IDs.
        track_id_name_map (dict, optional): Mapping of track IDs to names. Defaults to None.
        colors (list): List of colors for each class label.
        line_thickness (int, optional): Thickness of the bounding box lines. Defaults to 3.
        font_thickness (int, optional): Thickness of the label text. Defaults to 1.
        font_scale (float, optional): Scale factor for the label text size. Defaults to 0.5.

    Returns:
        numpy.ndarray: The annotated frame/image.
    """
    if track_id_name_map is None:
        track_id_name_map = {}
    annotator = Annotator(frame, line_width=line_thickness, font_size=font_scale)
    for result in results:
        boxes = result.boxes
        for box, track_id in zip(boxes, track_ids):
            b = box.xyxy[0] # get box coordinates in (left, top, right, bottom) format
            name = track_id_name_map.get(track_id) if track_id in track_id_name_map else None
            # Extract name from Path
            name = str(name).split("/")[-1] if name else None

            if name:
                annotator.box_label(b, name, color=colors)
            else:
                annotator.box_label(b, f"Track ID: {track_id}", color=colors)
    return annotator.result()

import datetime
import logging

def face_verification_thread(embed_index, images_path, results_queue, identify_queue, confidence_threshold=0.6, max_age_seconds=20):
    """
    Thread function for face verification.

    Args:
        embed_index (EmbeddingIndex): The embedding index used for face verification.
        images_path (List[str]): List of image paths used for identification.
        results_queue (Queue): Queue to receive results from the detection thread.
        identify_queue (Queue): Queue to send the identified track ID and name.
        confidence_threshold (float, optional): Confidence threshold for face verification. Defaults to 0.65.
        max_age_seconds (int, optional): Maximum age in seconds for a track ID. Defaults to 30.
    """
    track_id_name_map = {}
    track_id_score_map = {}  # To keep track of scores
    track_id_last_updated = {}  # To keep track of the last update time of each track ID

    while True:
        item = results_queue.get()
        if item is None:
            break
        frame, results, track_ids = item
        for result in results:
            pil_images = process_yolo_boxes(result, frame)
            for pil_image, track in zip(pil_images, track_ids):
                closest_index, score = run_face_verification(pil_image, embed_index)
                if closest_index is not None and score > confidence_threshold:
                    identified_name_path = images_path[closest_index]
                    identified_name = str(identified_name_path).split("/")[-1] if identified_name_path else None

                    current_time = datetime.datetime.now()
                    # Check if the track ID is too old
                    last_updated = track_id_last_updated.get(track_id)
                    if last_updated and (current_time - last_updated).total_seconds() > max_age_seconds:
                        # Consider the track ID as too old and reduce the score
                        track_id_score_map[track_id] = abs(track_id_score_map[track_id] - 0.15)  # Reduce the score for old track IDs
                        track_id_last_updated[track_id] = current_time # Update the last updated time
                        # Remove the track ID if the score is too low
                        if track_id_score_map[track_id] <= 0:
                            del track_id_name_map[track_id]
                            del track_id_score_map[track_id]
                            del track_id_last_updated[track_id]

                    # Update logic considering score and age
                    if track_id not in track_id_name_map or score > track_id_score_map.get(track_id, 0):
                        # Remove the old track ID if the name was previously associated with a different track ID
                        old_track_id = next((tid for tid, name in track_id_name_map.items() if name == identified_name), None)
                        if old_track_id:
                            del track_id_name_map[old_track_id]
                            del track_id_score_map[old_track_id]
                            del track_id_last_updated[old_track_id]
                        # Update the name, score, and last updated time for the current track ID
                        track_id_name_map[track_id] = identified_name
                        track_id_score_map[track_id] = score
                        track_id_last_updated[track_id] = current_time
                        logging.info(f"Identified: {identified_name} with id {track_id} and a score of {score}")

        identify_queue.put(track_id_name_map)

def identify_persons_in_video(video_file, model_file, visualize=False):
    embed_index, images_path = create_embedding(None)
    results_queue = queue.Queue(maxsize=1)
    identify_queue = queue.Queue(maxsize=3)

    detection_thread = threading.Thread(target=human_detection_thread, args=(video_file, model_file, results_queue, identify_queue, visualize))
    verification_thread = threading.Thread(target=face_verification_thread, args=(embed_index, images_path, results_queue, identify_queue))

    detection_thread.start()
    verification_thread.start()

    detection_thread.join()
    verification_thread.join()

if __name__ == "__main__":
    identify_persons_in_video(VIDEO_FILE_1, MODEL_FILE, visualize=True)
