# CNN Waste Classification with OpenCV and PyTorch

I am happy to announce that I have trained my own CNN for 50 epochs as a part of my learning journey. I used a Kaggle waste classification dataset (with modifications for 10 classes).  
This project represents a significant milestone in my learning journey with deep learning, where I developed a Convolutional Neural Network (CNN) to classify waste into 10 categories using OpenCV for image processing and PyTorch for model training and inference. Trained for 50 epochs on a modified Kaggle dataset, the model achieves a validation accuracy of 89.5%. This work explores the application of deep learning to real-world waste management, with innovations like real-time webcam predictions and detailed performance visualizations.

**The model predicts the type of waste from an image and can be used for smart recycling or educational applications.**


---

## Highlights

* **Customized Dataset:** Adapted the Kaggle Garbage Classification dataset to 10 specific waste categories, optimizing for balanced and practical classification.
* **Real-Time Prediction:** Leveraged OpenCV for seamless webcam and image-based predictions, enabling potential integration into automated waste sorting systems.
* **Robust CNN Architecture:** Designed a deep CNN with six convolutional layers, batch normalization, and dropout to ensure robust performance and prevent overfitting.
* **Comprehensive Evaluation:** Implemented a validation pipeline with a confusion matrix and accuracy graphs, providing clear insights into model performance.
* **Learning Journey:** This project combines knowledge from IBM’s Deep Learning with PyTorch course with hands-on implementation, reflecting my growth in understanding deep learning concepts.

---

## Table of Contents

* [Project Structure](#project-structure)
* [Setup & Installation](#setup--installation)
* [Dataset](#dataset)
* [Model Architecture](#model-architecture)
* [Training](#training)
* [Evaluation](#evaluation)
* [Usage Example](#usage-example)
* [Results](#results)
* [References](#references)
* [Author](#author)

---

## Project Structure

```
cnn-waste-classification-opencv-pytorch/
├── dataset/
│   ├── train/
│   │   ├── battery/
│   │   ├── cardboard/
│   │   ├── clothes/
│   │   ├── food_waste/
│   │   ├── glass/
│   │   ├── metal/
│   │   ├── paper/
│   │   ├── plastic/
│   │   ├── shoes/
│   │   ├── trash/
│   ├── val/
│   │   ├── battery/
│   │   ├── cardboard/
│   │   ├── clothes/
│   │   ├── food_waste/
│   │   ├── glass/
│   │   ├── metal/
│   │   ├── paper/
│   │   ├── plastic/
│   │   ├── shoes/
│   │   ├── trash/
├── saved_models/
│   └── best_model.pth
├── license
├── main.py
├── object-detection.py
├── validation-checker.py
├── validation-splitter.py
├── requirements.txt
```

* `main.py`: Trains the CNN model for 50 epochs and saves the best model to `saved_models/best_model.pth`.
* `object-detection.py`:carries out inference for garbage classification based on webcams or images and forecasts manual inspection.
* `validation-checker.py`: Evaluates the model, computes 89.56% accuracy, and generates a confusion matrix.
* `validation-splitter.py`: Splits the dataset into 80% training and 20% validation sets.
* `requirements.txt`: Lists dependencies like PyTorch, OpenCV, and NumPy.

---

## Setup & Installation

Follow these steps to set up and run the project:

```bash
git clone https://github.com/gokulseetharaman/cnn-waste-classification-opencv-pytorch.git
cd cnn-waste-classification-opencv-pytorch

# Create and activate a virtual environment:
python -m venv venv
venv\Scripts\activate
# On Linux: source venv/bin/activate

# Install dependencies:
pip install -r requirements.txt

# Train the model:
python3 main.py

# Perform predictions:
python3 object-detection.py

# Evaluate the model:
python3 validation-checker.py
```

---

## Dataset

The dataset is a modified version of the Kaggle Garbage Classification dataset, tailored to 10 waste categories:

Source : [Kaggle dataset](https://www.kaggle.com/datasets/mostafaabla/garbage-classification)

* Battery
* Cardboard
* Clothes
* Food Waste
* Glass
* Metal
* Paper
* Plastic
* Shoes
* Trash

The dataset is split into:

* **Training Set:** 80% of the data.
* **Validation Set:** 20% of the data.

Images are resized to 224x224 pixels to match the CNN’s input requirements.

---

## Model Architecture

The CNN is designed for efficiency and accuracy, with six convolutional layers followed by fully connected layers. Below is the model definition:

```python
import torch
import torch.nn as nn

class CNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(2,2),
            nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2,2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2,2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2,2),
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(), nn.MaxPool2d(2,2),
            nn.Conv2d(256, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(), nn.MaxPool2d(2,2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 3 * 3, 512),  # For 224x224 input
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x
```

**Summary Table:**

| Layer Type         | Count | Details                            |
| ------------------ | ----- | ---------------------------------- |
| Input “Layer”      | 1     | Conv2d(3, 16, kernel\_size=3, ...) |
| Conv Hidden Layers | 6     | Conv2d-BatchNorm2d-ReLU-MaxPool2d  |
| FC Hidden Layer    | 1     | Linear(512*3*3, 512)               |
| Output Layer       | 1     | Linear(512, num\_classes)          |

**Neuron Count**

* **Convolutional Layers:** Feature maps increase progressively (16, 32, 64, 128, 256, 512).
* **Fully Connected Layers:**

  * First FC: 512 × 3 × 3 = 4608 inputs → 512 outputs.
  * Output FC: 512 inputs → 10 outputs (num\_classes).

**Input Processing**

* Input: 224x224 RGB images.
* After six MaxPool2d(2,2) layers: 224 → 112 → 56 → 28 → 14 → 7 → 3 (rounded down).

---

## Training

The model was trained with the following hyperparameters:

* Number of Classes: 10
* Batch Size: 8
* Learning Rate: 5e-4
* Epochs: 50
* Early Stopping Patience: 10
* Optimizer: AdamW
* Loss Function: Cross Entropy Loss

Run `main.py` to train the model for 50 epochs. The best model is saved to `saved_models/best_model.pth`.

---

## Evaluation

The model was evaluated using `validation-checker.py`, which:

* Achieves a validation accuracy of **89.56%**.
* Generates a confusion matrix to analyze classification performance across categories.

Predictions are supported for both webcam feeds and static images via `object-detection.py`.


---

## Usage Example

**Train the Model:**

```bash
python3 main.py
```

**Predict with Webcam or Image:**

```bash
python3 object-detection.py
```

**Example:**
Below is the updated `predict_frame` function from `object-detection.py`, addressing the type mismatch error (Input type (double) and bias type (float) should be the same):

```python
def predict_frame(self, frame):
    try:
        import cv2
        import numpy as np
        import torch
        
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (224, 224))
        cv2.imshow("prediction", img)
        
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)  # Ensure float32
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)   # Ensure float32
        img = img.astype(np.float32) / 255.0
        img = (img - mean) / std
        img = np.transpose(img, (2, 0, 1))  # HWC to CHW
        img_tensor = torch.from_numpy(img).unsqueeze(0).float().to(self.device)  # Ensure float32
        
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
```

**Fix Explanation:** The error was resolved by ensuring the input tensor and model parameters use float32 precision. The mean and std arrays are set to np.float32, and `.float()` is explicitly applied to `img_tensor`.

**Example output:**

```
Predicted Class: Plastic
Confidence: 0.92
```

**Evaluate the Model:**

```bash
python3 validation-checker.py
```

Outputs:

* Validation accuracy: 89.5%
* Confusion matrix


---

## Results

The model achieved a validation accuracy of **89.5%**. Below are the detailed precision, recall, F1-score, and support for each class as calculated on the validation set:

<img width="527" alt="image" src="https://github.com/user-attachments/assets/7e146583-880b-4fda-ab44-688ff2a137f2" />


**Visualizations:**

* Confusion Matrix: Shows classification performance across the 10 classes.
  
    ![confusion matrix](https://github.com/user-attachments/assets/733887d2-2385-41c3-9415-cae06602f19d)
* Webcam Prediction: Real-time classification from webcam feed.
  
  <img width="949" alt="Screenshot 2025-05-26 201547" src="https://github.com/user-attachments/assets/edf863a2-f185-439c-ada8-eb8915c78142" />
* Image Prediction: Accurate classification of static images.
  
  <img width="164" alt="image" src="https://github.com/user-attachments/assets/6f7bd08c-b139-412c-b1e2-890eb618e075" />


---

## References

* IBM Deep Learning with PyTorch Course
* Gyawali, D., Regmi, A., Shakya, A., Gautam, A., and Shrestha, S., 2020. Comparative analysis of multiple deep CNN models for waste classification. arXiv preprint arXiv:2004.02168.
* Kaggle Garbage Classification Dataset
