import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import cv2
import os
import numpy as np
import time
import random

class CNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(256, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(), nn.MaxPool2d(2, 2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 3 * 3, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


class WebcamPredictor:
    def __init__(self, model_path):
        # Check for CUDA availability
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        # Load the saved model
        checkpoint = torch.load(model_path, map_location=self.device, weights_only= True)
        self.class_names = checkpoint['class_names']

        # Initialize model
        self.model = CNN(num_classes=len(self.class_names))
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()

        # Define image transformations
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def predict_frame(self, frame):
        try:
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (224, 224))

            cv2.imshow("prediction", img)
            mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
            std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
            img = img.astype(np.float32) / 255.0
            img = (img - mean) / std
            img = np.transpose(img, (2, 0, 1))
            img_tensor = torch.from_numpy(img).unsqueeze(0).float().to(self.device)
            # Predict
            with torch.no_grad():
                output = self.model(img_tensor)
                probabilities = torch.nn.functional.softmax(output, dim=1)
                predicted_idx = torch.argmax(probabilities, dim=1).item()
                confidence = probabilities[0][predicted_idx].item()
                predicted_class = self.class_names[predicted_idx]

            return predicted_class, confidence

        except Exception as e:
            print(f"Error during prediction: {str(e)}")
            return None, None

    def start_webcam(self):
        # Initialize webcam
        cap = cv2.VideoCapture(0)

        # Check if webcam is opened successfully
        if not cap.isOpened():
            print("Error: Could not open webcam")
            return

        # Set webcam properties (optional)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        print("Webcam started. Press 'q' to quit.")

        # Variables for FPS calculation
        fps_start_time = time.time()
        fps_frame_count = 0
        fps = 0

        while True:
            # Read frame from webcam
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read frame")
                break

            # Calculate FPS
            fps_frame_count += 1
            if fps_frame_count >= 30:
                fps = fps_frame_count / (time.time() - fps_start_time)
                fps_start_time = time.time()
                fps_frame_count = 0

            # Make prediction
            predicted_class, confidence = self.predict_frame(frame)

            # Draw prediction and FPS on frame
            if predicted_class is not None:
                # Draw semi-transparent background for text
                overlay = frame.copy()
                cv2.rectangle(overlay, (10, 10), (400, 90), (0, 0, 0), -1)
                alpha = 0.6
                frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

                # Draw text
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(frame, f"Class: {predicted_class}", (20, 40), font, 0.8, (0, 255, 0), 2)
                cv2.putText(frame, f"Confidence: {confidence:.2%}", (20, 70), font, 0.8, (0, 255, 0), 2)
                cv2.putText(frame, f"FPS: {fps:.1f}", (20, 100), font, 0.8, (0, 255, 0), 2)

            # Display frame
            cv2.imshow('Webcam Prediction', frame)

            # Check for 'q' key to quit
            if cv2.waitKey(30) & 0xFF == ord('q'):
                break

        # Release resources
        cap.release()
        cv2.destroyAllWindows()


def main():
    # Model path - update this to your model's path
    model_path = "saved_models/best_model.pth"

    # Initialize predictor and start webcam
    predictor = WebcamPredictor(model_path)
    predictor.start_webcam()


if __name__ == "__main__":
    main()