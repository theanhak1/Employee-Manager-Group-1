import sys
import cv2
import numpy as np
sys.path.append('./')
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.init as init
from torchvision.transforms import transforms
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch.utils.data as data_utils
from ultralytics import YOLO

# Load the YOLOv8n-pose model
yolo_model = YOLO('yolov8n-pose.pt').to('cuda' if torch.cuda.is_available() else 'cpu')

def preprocess_keypoints(results):
    if results[0].keypoints.xyn.cpu().numpy() is None:
        return None
    kp = results[0].keypoints.xyn.cpu().numpy()  # Get the (x, y, confidence) values for each keypoint
    keypoints = torch.tensor(kp, dtype=torch.float32)
    # Get the size of the first dimension
    first_dim_size = keypoints.size(0)

    # Flatten all dimensions after the first one while keeping the first dimension intact
    flattened_tensor = keypoints.view(first_dim_size, -1)
    return flattened_tensor

# Define the complex dense model
class ComplexDenseClassifier(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        super(ComplexDenseClassifier, self).__init__()
        self.fc_layers = nn.ModuleList()
        prev_size = input_size

        for hidden_size in hidden_sizes:
            self.fc_layers.append(nn.Linear(prev_size, hidden_size))
            self.fc_layers.append(nn.BatchNorm1d(hidden_size))
            self.fc_layers.append(nn.Dropout(0.35))
            self.fc_layers.append(nn.ReLU())
            prev_size = hidden_size

        self.fc_layers.append(nn.Linear(prev_size, output_size))

    def forward(self, x):
        out = x
        for layer in self.fc_layers:
            out = layer(out)
        return out
    
    @staticmethod
    def get_trainable_params(model):
        trainable_params = filter(lambda p: p.requires_grad, model.parameters())
        return sum([torch.numel(p) for p in trainable_params])

def initialize_weights(model, init_method):
    for module in model.modules():
        if isinstance(module, nn.Linear):
            init_method(module.weight)

if __name__ == "__main__":
    is_cuda = torch.cuda.is_available()

    # Hyperparameters
    input_size = 17 * 2
    hidden_sizes = [64, 512, 256, 64]
    output_size = 1
    learning_rate = 0.0001
    num_epochs = 100
    batch_size = 32

    # Create an instance of the dense model
    model = ComplexDenseClassifier(input_size, hidden_sizes, output_size).to('cuda' if is_cuda else 'cpu')
    print(model.get_trainable_params(model))
    initialize_weights(model, init_method=init.kaiming_normal_)

    # Define the loss function and optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.3)
    lr_scheduler = CosineAnnealingLR(optimizer, T_max=10, eta_min=0.00001, last_epoch=-1)

    from greet_detection.data.dataloader import get_dataloader

    # Load the dataloader
    dataloaders = get_dataloader(["greet_detection/data/greets", "greet_detection/data/not_greets"],
                                      batch_size=batch_size, transform_list=[
                                        transforms.RandomHorizontalFlip(), 
                                        transforms.RandomRotation(20),
                                        ],
                                      even_load=True,
                                      example_plot=False,
                                      train_percentage=0.9)
    train_dataloader = dataloaders['train']
    eval_dataloader = dataloaders['eval']

    import matplotlib.pyplot as plt

    # Training loop
    total_loss = 0
    total_batches = 0
    losses = []
    eval_losses = []

    for epoch in range(num_epochs):
        model.train()
        for inputs, labels in train_dataloader:
            inputs, labels = inputs.to('cuda' if is_cuda else 'cpu'), labels.to('cuda' if is_cuda else 'cpu')
            # Run inference on the resized images using YOLOv8n-pose
            results_batch = yolo_model(inputs, verbose=False)

            filtered_results = []  # List to hold results with detected persons
            filtered_labels = []   # List to hold corresponding labels

            for result, label in zip(results_batch, labels):
                if len(result) > 0 and result[0].keypoints.xyn.cpu().numpy() is not None:
                    # If at least one person is detected, add result and label to filtered lists
                    filtered_results.append(result)
                    filtered_labels.append(label)

            if not filtered_results:
                continue  # Skip this batch if no person is detected in any example

            # Preprocess the keypoints
            keypoints_batch = [preprocess_keypoints(results) for results in filtered_results]
            keypoints_batch = torch.cat(keypoints_batch, dim=0)

            keypoints_batch, filtered_labels = keypoints_batch.to('cuda' if is_cuda else 'cpu'), torch.tensor(filtered_labels).to('cuda' if is_cuda else 'cpu')
            optimizer.zero_grad()
            outputs = model(keypoints_batch)
            loss = criterion(outputs.squeeze(), filtered_labels.float())
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_batches += 1

            losses.append(loss.item())
            lr_scheduler.step()

        # Evaluation loop
        model.eval()
        eval_batch_losses = []
        with torch.no_grad():
            for inputs, labels in eval_dataloader:
                inputs, labels = inputs.to('cuda' if is_cuda else 'cpu'), labels.to('cuda' if is_cuda else 'cpu')
                # Run inference on the images using YOLOv8n-pose
                results_batch = yolo_model(inputs)

                filtered_results = []  # List to hold results with detected persons
                filtered_labels = []   # List to hold corresponding labels

                for result, label in zip(results_batch, labels):
                    if len(result) > 0 and result[0].keypoints.xyn.cpu().numpy() is not None:
                        # If at least one person is detected, add result and label to filtered lists
                        filtered_results.append(result)
                        filtered_labels.append(label)

                if not filtered_results:
                    continue  # Skip this batch if no person is detected in any example

                # Preprocess the keypoints
                keypoints_batch = [preprocess_keypoints(results) for results in filtered_results]
                keypoints_batch = torch.cat(keypoints_batch, dim=0)

                keypoints_batch, filtered_labels = keypoints_batch.to('cuda' if is_cuda else 'cpu'), torch.tensor(filtered_labels).to('cuda' if is_cuda else 'cpu')
                outputs = model(keypoints_batch)
                eval_loss = criterion(outputs.squeeze(), filtered_labels.float())
                eval_batch_losses.append(eval_loss.item())

            eval_losses.append(sum(eval_batch_losses) / len(eval_batch_losses))
            average_loss = total_loss / total_batches
            print(f'Epoch [{epoch + 1}/{num_epochs}], Average Loss: {average_loss}, Average Eval Loss: {eval_losses[-1]}')

    # Plot the loss and the eval loss
    plt.plot(losses, label='Train Loss')
    plt.plot(eval_losses, label='Eval Loss')
    plt.xlabel('Steps')
    plt.ylabel('Loss')
    plt.title('Loss by Steps')
    plt.legend()
    plt.savefig('greet_detection/checkpoints/loss_plot.png')

    # Save the trained model
    torch.save(model.state_dict(), 'greet_detection/checkpoints/greeting_model.pth')
