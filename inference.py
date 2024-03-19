import sys
sys.path.append('./')
import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from greet_detection.model import ComplexDenseClassifier, preprocess_keypoints, yolo_model
from greet_detection.data import get_all_paths, ResizeAndPad


USE_CUDA = True
is_cuda = torch.cuda.is_available()
resize_and_pad = ResizeAndPad(384, 384)

# if __name__ == "__main__":
# Hyperparameters
input_size = 17 * 2
hidden_sizes = [64, 512, 256, 64]
output_size = 1

# Load the model outside of the scripted function
model_path = 'greet_detection/checkpoints/greeting_model.pth'
model = ComplexDenseClassifier(input_size, hidden_sizes, output_size).to('cuda' if is_cuda and USE_CUDA else 'cpu')
model.load_state_dict(torch.load(model_path))
model.eval()

def scripted_predict_greeting(img_tensor: torch.Tensor) -> float:
    with torch.no_grad():
        results_batch = yolo_model(img_tensor)
        # Preprocess the keypoints
        keypoints_batch = [preprocess_keypoints(results) for results in results_batch]
        keypoints_batch = torch.cat(keypoints_batch, dim=0)

        keypoints_batch = keypoints_batch.to('cuda' if is_cuda and USE_CUDA else 'cpu')

        output = model(keypoints_batch)
    probability = torch.sigmoid(output).item()
    return probability

def load_and_preprocess_image(image):
    transform = transforms.Compose([
        resize_and_pad,
        transforms.ToTensor()
    ])
    img = Image.open(image).convert('RGB') if isinstance(image, str) else image
    # Save a unique image for each inference
    # img.save(f'greet_detection/images/{str(time.time())}.jpg')
    img_tensor = transform(img).unsqueeze(0)  # Add batch dimension
    return img_tensor

import time

def perform_greeting_inference(image_dir: str, threshold: float = 0.5, verbose: bool=True):
    image_paths = get_all_paths(image_dir, ["jpg", "png", "jpeg", "JPG", "PNG", "JPEG"]) if isinstance(image_dir, str) else image_dir
    results = []
    start_time = time.time()
    for image_path in image_paths:
        # Example usage for inference
        img = load_and_preprocess_image(image_path)
        img_tensor = img.to('cuda' if is_cuda and USE_CUDA else 'cpu')
        prediction = scripted_predict_greeting(img_tensor)

        # Define a threshold for classification
        threshold = 0.5
        result =  prediction > threshold # True if greeting, False if not
        if verbose: print(f"Prediction: {prediction} for {image_path}, Result: {'''Greetings!''' if result else '''Not a greeting.'''}")
        results.append(result)

    end_time = time.time()
    execution_time = end_time - start_time
    if verbose: print(f"Execution time: {execution_time} seconds")

    return results



if __name__ == "__main__":

    image_dir = 'greet_detection/images/'
    perform_greeting_inference(image_dir, verbose=True)
