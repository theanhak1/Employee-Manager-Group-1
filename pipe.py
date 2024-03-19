import datetime
import sys
import os
import time
import logging
import threading
import queue
import cv2
sys.path.append('./')
from ultralytics.utils.plotting import Annotator
from face_verification import run_face_verification, create_embedding
from human_detection import process_video, process_yolo_boxes, initialize_model, initialize_video_capture, release_video
from greet_detection import perform_greeting_inference
from dress_verification import perform_dress_verification

# Configuration Constants
CAMERA_ID = 0
MODEL_FILE = 'yolov8n.pt'
DEVICE_ID = "0"

# Configure logging
logging.basicConfig(filename='identification_log.txt', level=logging.INFO, format='%(asctime)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

def human_detection_thread(video_file, model_file, results_queue, identify_queue, visualize=False, width=1000, height=None, process_interval=2):
    """
    Thread function for human detection.

    Args:
        video_file (str): Path to the video file.
        model_file (str): Path to the model file.
        results_queue (Queue): Queue to store the processed video frames and results.
        identify_queue (Queue): Queue to retrieve identification results.
        visualize (bool, optional): Whether to visualize the detection results. Defaults to False.
        width (int, optional): Width of the video frame. Defaults to 320.
        height (int, optional): Height of the video frame. Defaults to None.
        process_interval (int, optional): Interval between processing video frames. Defaults to 2.
    """
    model = initialize_model(model_file)
    video, _, _ = initialize_video_capture(video_file, width, height)
    track_id_name_map = {}
    start_time = time.time()
    while True:
        ret, frame, results = process_video(video, model, 0, visualize=False, confidence=0.7, verbose=False)
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

def plot_results(results, frame, track_ids, track_id_name_map=None, colors=(0, 0, 255), line_thickness=1, font_scale=0.1):
    """
    Plot the results on the given frame.

    Args:
        results (list): List of results containing boxes.
        frame (numpy.ndarray): The frame to plot the results on.
        track_ids (list): List of track IDs corresponding to the boxes.
        track_id_name_map (dict, optional): Mapping of track IDs to names. Defaults to None.
        colors (tuple, optional): Color of the box labels. Defaults to (0, 0, 255) (blue).
        line_thickness (int, optional): Thickness of the box lines. Defaults to 1.
        font_scale (float, optional): Scale of the font for the box labels. Defaults to 0.1.

    Returns:
        numpy.ndarray: The frame with the plotted results.
    """
    if track_id_name_map is None:
        track_id_name_map = {}
    annotator = Annotator(frame, line_width=line_thickness, font_size=font_scale)
    for result in results:
        boxes = result.boxes
        for box, track_id in zip(boxes, track_ids):
            b = box.xyxy[0] # get box coordinates in (left, top, right, bottom) format
            e_info = track_id_name_map.get(track_id) if track_id in track_id_name_map else None
            name = e_info['name'] if e_info else None
            # Extract name from Path
            name = os.path.basename(name) if name else None
            # name = str(name).split("/")[-1].split(".")[0] if name else None

            if name:
                annotator.box_label(b, f"Name: {name} Dress: {e_info['correct_dress']} Greeted: {e_info['has_greeted']} TrID: {track_id}", color=colors)
            else:
                annotator.box_label(b, f"Track ID: {track_id}", color=colors)
    return annotator.result()


def face_verification_dress_and_greeting_thread(embed_index, images_path, results_queue, identify_queue, confidence_threshold=0.65, max_age_seconds=10, cooldown_seconds=5):
    """
    Thread function for face verification, dress verification, and greeting detection.

    Args:
        embed_index (EmbeddingIndex): The embedding index for face verification.
        images_path (List[str]): List of image paths for identification.
        results_queue (Queue): Queue for receiving frame results.
        identify_queue (Queue): Queue for sending identification results.
        confidence_threshold (float, optional): Confidence threshold for face verification. Defaults to 0.65.
        max_age_seconds (int, optional): Maximum age of a track ID in seconds. Defaults to 10.
        cooldown_seconds (int, optional): Cooldown period for greeting detection in seconds. Defaults to 5.
    """
    track_id_name_map = {}
    track_id_score_map = {}  # To keep track of scores
    track_id_last_updated = {}  # To keep track of the last update time of each track ID
    last_greeting_time = None

    while True:
        item = results_queue.get()
        if item is None:
            break
        frame, results, track_ids = item
        for result in results:
            pil_images = process_yolo_boxes(result, frame)
            for pil_image, track_id in zip(pil_images, track_ids):
                closest_index, score = run_face_verification(pil_image, embed_index)
                current_time = datetime.datetime.now()
                if closest_index is not None and score > confidence_threshold:
                    identified_name_path = images_path[closest_index]
                    identified_name = str(identified_name_path).split("/")[-1] if identified_name_path else None

                    # Update logic considering score and age
                    if track_id not in track_id_name_map or score > track_id_score_map.get(track_id, 0):
                        # Remove the old track ID if the name was previously associated with a different track ID
                        old_track_id = next((tid for tid, e_info in track_id_name_map.items() if e_info['name'] == identified_name), None)
                        if old_track_id:
                            del track_id_name_map[old_track_id]
                            del track_id_score_map[old_track_id]
                            del track_id_last_updated[old_track_id]
                        # Update the name, score, and last updated time for the current track ID
                        correct_dress = perform_dress_verification(pil_image) == "correct"
                        if correct_dress:  
                            logging.info("Correct dress detected at {} for identified employee {}".format(current_time, identified_name))
                        else:
                            logging.info("Incorrect dress detected at {} for identified employee {}".format(current_time, identified_name))
                        track_id_name_map[track_id] = {"name": identified_name, "correct_dress": correct_dress, "has_greeted": False}
                        track_id_score_map[track_id] = score
                        track_id_last_updated[track_id] = current_time
                        logging.info(f"Identified: {identified_name} with id {track_id} and a score of {score}")
                
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

                # Perform greeting detection for identified employee
                if any(track_id_other not in track_id_name_map for track_id_other in set(track_ids) - {track_id}):
                    e_info = track_id_name_map.get(track_id)
                    # Check if employee is in correct dress
                    if e_info and e_info['correct_dress']:
                        if last_greeting_time is None or (current_time - last_greeting_time).total_seconds() > cooldown_seconds:
                            greeting_detected = perform_greeting_inference([pil_image], threshold=0.65)[0] 
                            if greeting_detected:
                                last_greeting_time = current_time
                                track_id_name_map[track_id]['has_greeted'] = True
                                logging.info("Greeting detected at {} for identified employee {}".format(current_time, e_info['name']))
                            else:
                                track_id_name_map[track_id]['has_greeted'] = False

        identify_queue.put(track_id_name_map)

def identify_persons_in_video(video_file, model_file, visualize=False):
    embed_index, images_path = create_embedding(None)
    results_queue = queue.Queue(maxsize=1)
    identify_queue = queue.Queue(maxsize=3)

    detection_thread = threading.Thread(target=human_detection_thread, args=(video_file, model_file, results_queue, identify_queue, visualize))
    verification_and_greeting_thread = threading.Thread(target=face_verification_dress_and_greeting_thread, args=(embed_index, images_path, results_queue, identify_queue))

    detection_thread.start()
    verification_and_greeting_thread.start()

    detection_thread.join()
    verification_and_greeting_thread.join()

if __name__ == "__main__":
    identify_persons_in_video(CAMERA_ID, MODEL_FILE, visualize=True)