import torch
import os
import argparse
import cv2
import shutil
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from PIL import Image


def load_model(weight_path="model.safetensors"):
    """
    Load the model and its weights from the specified path.

    Parameters:
    weight_path (str): Path to the model weights file.

    Returns:
    model (torch.nn.Module): Loaded model.
    device (torch.device): Device on which the model is loaded.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.resnet18(weights=None).to(device)

    if os.path.exists(weight_path):
        model.load_state_dict(torch.load(weight_path, map_location=device))
        model.eval()
    else:
        print("Model weight file not found. Please train the model first using --train.")
        exit()

    return model, device


def train_model(dataset_path="images", weight_path="model.safetensors", epochs=5, batch_size=16):
    """
    Train the model on the dataset.

    Parameters:
    dataset_path (str): Path to the dataset folder.
    weight_path (str): Path to save the model weights.
    epochs (int): Number of training epochs.
    batch_size (int): Batch size for training.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    criterion = torch.nn.CrossEntropyLoss()
    transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])

    dataset = datasets.ImageFolder(root=dataset_path, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        accuracy = 100 * correct / total
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss:.4f}, Accuracy: {accuracy:.2f}%")

    torch.save(model.state_dict(), weight_path)
    print("Training complete. Weights saved to model.safetensors")


def predict(image_path, model, device):
    """
    Predict whether the given image is AI-generated or original.

    Parameters:
    image_path (str): Path to the image file.
    model (torch.nn.Module): Loaded model.
    device (torch.device): Device on which the model is loaded.

    Returns:
    str: Prediction result ("AI-generated" or "Original").
    """
    if not os.path.exists(image_path):
        print(f"Error: Image '{image_path}' not found.")
        return "Error"

    transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image)

    predicted_label = torch.argmax(outputs, dim=-1).item()

    return "AI-generated" if predicted_label == 1 else "Original"


def process_video(video_path, model, device):
    """
    Process a video and detect AI-generated frames.

    Parameters:
    video_path (str): Path to the video file.
    model (torch.nn.Module): Loaded model.
    device (torch.device): Device on which the model is loaded.
    """
    if not os.path.exists(video_path):
        print(f"Error: Video '{video_path}' not found.")
        return

    temp_folder = "temp_frames"
    os.makedirs(temp_folder, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_path = os.path.join(temp_folder, f"frame_{frame_count}.jpg")
        cv2.imwrite(frame_path, frame)
        frame_count += 1
    cap.release()

    print("Processing frames for AI detection...")
    for frame in os.listdir(temp_folder):
        frame_path = os.path.join(temp_folder, frame)
        label = predict(frame_path, model, device)
        if label == "AI-generated":
            print(f"AI-generated content detected in frame: {frame}")
            break

    shutil.rmtree(temp_folder)
    print("Video analysis complete.")


def main():
    """
    Main function to parse command-line arguments and execute the corresponding function.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true", help="Train the model with images in the dataset folder")
    parser.add_argument("--image", type=str, help="Path to the image for detection")
    parser.add_argument("--video", type=str, help="Path to the video for AI detection")
    args = parser.parse_args()

    # Display system and GPU information
    print(f"CUDA Available: {torch.cuda.is_available()}")
    print(f"Number of GPUs: {torch.cuda.device_count()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("No GPU detected")

    if args.train:
        train_model()
    elif args.image:
        model, device = load_model()
        label = predict(args.image, model, device)
        print(f"Prediction: {label}")
    elif args.video:
        model, device = load_model()
        process_video(args.video, model, device)
    else:
        print(
            "Please provide --train to train the model, --image <image_path> for image detection, or --video <video_path> for video detection.")


if __name__ == "__main__":
    main()