import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision.models import resnet18
from PIL import Image

# Define device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define model path
model_path = "dress_verification/checkpoints/cloth_classification.pth"

# Load pre-trained ResNet18 model
def load_model():
    model = resnet18(pretrained=False)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2)
    model.load_state_dict(torch.load(model_path))
    return model.to(device)

# Preprocess image
def preprocess_image(image_path):
    image = Image.open(image_path).convert('RGB') if isinstance(image_path, str) else image_path
    image.save("dress_verification/processed_image.jpg")
    transform = transforms.Compose([
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    input_image = transform(image)
    return input_image.unsqueeze(0).to(device)

# Perform inference
def predict_image_class(model, input_image):
    model.eval()
    with torch.no_grad():
        output = model(input_image)
    probabilities = F.softmax(output, dim=1)
    predicted_class = torch.argmax(probabilities, dim=1).item()
    return predicted_class

# Get class label
def get_class_label(predicted_class):
    class_labels = ['correct', 'incorrect']
    return class_labels[predicted_class]

def perform_dress_verification(image_path) -> str:
    # Load model
    model = load_model()

    # Load and preprocess the image
    input_image = preprocess_image(image_path)

    # Perform inference
    predicted_class = predict_image_class(model, input_image)

    # Get the class label
    predicted_label = get_class_label(predicted_class)

    return predicted_label

if __name__ == "__main__":
    perform_dress_verification()