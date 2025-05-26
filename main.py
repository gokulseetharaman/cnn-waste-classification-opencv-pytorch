import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets, models
from torch.utils.data import DataLoader
import os

class CNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(2,2),
            nn.Conv2d(16, 32, 3, padding=1),nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2,2),
            nn.Conv2d(32, 64, 3, padding=1),nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2,2),
            nn.Conv2d(64, 128, 3, padding=1),nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2,2),
            nn.Conv2d(128, 256, 3, padding=1),nn.BatchNorm2d(256), nn.ReLU(), nn.MaxPool2d(2,2),
            nn.Conv2d(256, 512, 3, padding=1),nn.BatchNorm2d(512), nn.ReLU(), nn.MaxPool2d(2,2),
        )
        # Calculate the output size after all pools for your image size (e.g., 224x224)
        # For 224x224 and 6 MaxPool2d(2,2): 224 -> 112 -> 56 -> 28 -> 14 -> 7 -> 3 (rounded down)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 3 * 3, 512),  # If input is 224x224
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


class Dataset:
    def __init__(self, train_dir, val_dir, batch_size):
        self.train_dir = train_dir
        self.val_dir = val_dir
        self.batch_size = batch_size

        self.train_transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.3),
            transforms.RandomRotation(degrees=45),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
            transforms.RandomAffine(degrees=20, translate=(0.1, 0.1), scale=(0.8, 1.2)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        self.val_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def get_loaders(self):
        train_dataset = datasets.ImageFolder(self.train_dir, transform=self.train_transform)
        val_dataset = datasets.ImageFolder(self.val_dir, transform=self.val_transform)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=os.cpu_count(), pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=os.cpu_count(), pin_memory=True)
        return train_loader, val_loader, train_dataset.classes

# 3️⃣ Trainer Class


class Trainer:
    def __init__(self, model, device, lr, patience):
        self.model = model
        self.device = device
        self.patience = patience
        self.criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        self.optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        self.scheduler = None
        self.best_accuracy = 0.0
        self.early_stopping_counter = 0

    def set_scheduler(self, train_loader, num_epochs):
        self.scheduler = optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=self.optimizer.param_groups[0]['lr'],
            epochs=num_epochs,
            steps_per_epoch=len(train_loader),
            pct_start=0.3,
            anneal_strategy='cos'
        )

    def train_one_epoch(self, train_loader):
        self.model.train()
        total_loss, total_correct, total_samples = 0, 0, 0
        for data, targets in train_loader:
            data, targets = data.to(self.device), targets.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(data)
            loss = self.criterion(outputs, targets)
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            if self.scheduler is not None:
                self.scheduler.step()
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total_correct += predicted.eq(targets).sum().item()
            total_samples += targets.size(0)
        avg_loss = total_loss / len(train_loader)
        accuracy = 100. * total_correct / total_samples
        return avg_loss, accuracy

    def train(self, train_loader, val_loader, num_epochs, validator, saver, class_names):
        self.set_scheduler(train_loader, num_epochs)
        for epoch in range(num_epochs):
            print(f"\nEpoch [{epoch+1}/{num_epochs}]")
            train_loss, train_acc = self.train_one_epoch(train_loader)
            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            val_loss, val_acc = validator.validate(val_loader, self.model, self.device, self.criterion)
            print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")

            # Save best model
            if val_acc > self.best_accuracy:
                self.best_accuracy = val_acc
                saver.save(self.model, self.optimizer, epoch, self.best_accuracy, class_names)
                print(f"New best model saved with accuracy: {self.best_accuracy:.2f}%")
                self.early_stopping_counter = 0
            else:
                self.early_stopping_counter += 1

            if self.early_stopping_counter >= self.patience:
                print(f"Early stopping triggered. Best accuracy: {self.best_accuracy:.2f}%")
                break

        print(f"Training completed. Best validation accuracy: {self.best_accuracy:.2f}%")


class Validation:
    def __init__(self):
        pass

    def validate(self, val_loader, model, device, criterion):
        model.eval()
        total_loss, total_correct, total_samples = 0, 0, 0
        with torch.no_grad():
            for data, targets in val_loader:
                data, targets = data.to(device), targets.to(device)
                outputs = model(data)
                loss = criterion(outputs, targets)
                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total_correct += predicted.eq(targets).sum().item()
                total_samples += targets.size(0)
        avg_loss = total_loss / len(val_loader)
        accuracy = 100. * total_correct / total_samples
        return avg_loss, accuracy


class ModelSaver:
    @staticmethod
    def save(model, optimizer, epoch, best_accuracy, class_names, save_path="saved_models/best_model.pth"):
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_accuracy': best_accuracy,
            'class_names': class_names
        }, save_path)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = 10
    batch_size = 8
    learning_rate = 5e-4
    num_epoch = 50
    patience = 10
    train_dir = "dataset/train"
    val_dir =  "dataset/val"

    print("Device being used:", device)

    # Dataset and loaders
    data_loader = Dataset(train_dir, val_dir, batch_size)
    train_loader, val_loader, class_names = data_loader.get_loaders()

    # Model
    model = CNN(num_classes=num_classes)
    model = model.to(device)

    # Validators and savers
    validator = Validation()
    saver = ModelSaver()

    # Trainer
    trainer = Trainer(model, device, learning_rate, patience)
    trainer.train(train_loader, val_loader, num_epoch, validator, saver, class_names)

if __name__ == "__main__":
    main()
