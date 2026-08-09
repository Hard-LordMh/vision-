import json
import os
import csv
from typing import Any, cast

import torch
from torch import nn, optim
from torchvision import datasets, models, transforms
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from app.database.database import SessionLocal
from app.models.schema import ModelInfo, TrainingMetric, TrainingRun
from app.services.dataset_service import get_dataset_processed_dir
from app.ml.tabular_ann import TabularANN

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "storage", "models")
os.makedirs(MODELS_DIR, exist_ok=True)

class TrainingEngine:
    def __init__(self, run_id: int):
        self.run_id = run_id
        self.db = SessionLocal()
        self.run: Any = self.db.query(TrainingRun).filter(TrainingRun.id == run_id).first()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    def start_training(self):
        if not self.run:
            print(f"Training run {self.run_id} not found")
            return
            
        try:
            self.run.status = "TRAINING"
            self.db.commit()
            
            dataset_dir = get_dataset_processed_dir(self.run.dataset_id)
            csv_path = os.path.join(dataset_dir, "data.csv")
            
            if os.path.exists(csv_path):
                self._train_tabular(csv_path)
            else:
                self._train_vision(dataset_dir)
                
        except Exception as e:
            if self.run:
                self.run.status = "FAILED"
                self.db.commit()
            print(f"Training failed: {e}")
        finally:
            self.db.close()

    def _train_tabular(self, csv_path: str):
        # 1. Read CSV using csv.DictReader
        rows = []
        with open(csv_path, mode='r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append(row)
                
        # 2. Preprocess (remove missing values)
        valid_rows = []
        for r in rows:
            if any(v.strip() == "" or v is None for v in r.values()):
                continue
            valid_rows.append(r)
            
        if not valid_rows:
            raise ValueError("No valid rows left in tabular dataset after removing missing values")
            
        # 3. Encode categorical data
        X = []
        y = []
        
        for r in valid_rows:
            internship_val = r.get("Internship", r.get("internship", "No"))
            internship_encoded = 1.0 if internship_val.lower() == "yes" else 0.0
            
            cgpa = float(r.get("CGPA", r.get("cgpa", 0.0)))
            aptitude = float(r.get("Aptitude score", r.get("aptitude_score", r.get("aptitude", 0.0))))
            comm = float(r.get("Communication skills", r.get("communication_skills", r.get("communication", 0.0))))
            coding = float(r.get("Coding skills", r.get("coding_skills", r.get("coding", 0.0))))
            projects = float(r.get("Projects completed", r.get("projects_completed", r.get("projects", 0.0))))
            
            placement_val = r.get("Placement", r.get("placement", "No"))
            placement_encoded = 1 if placement_val.lower() == "yes" else 0
            
            X.append([cgpa, aptitude, comm, coding, internship_encoded, projects])
            y.append(placement_encoded)
            
        # 4. Split into training and testing sets
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # 5. Normalize/scale the data
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        
        X_train_t = torch.tensor(X_train_scaled, dtype=torch.float32).to(self.device)
        y_train_t = torch.tensor(y_train, dtype=torch.long).to(self.device)
        X_val_t = torch.tensor(X_val_scaled, dtype=torch.float32).to(self.device)
        y_val_t = torch.tensor(y_val, dtype=torch.long).to(self.device)
        
        # Dataloaders
        from torch.utils.data import TensorDataset, DataLoader
        train_ds = TensorDataset(X_train_t, y_train_t)
        val_ds = TensorDataset(X_val_t, y_val_t)
        
        train_loader = DataLoader(train_ds, batch_size=self.run.batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=self.run.batch_size, shuffle=False)
        
        # Setup Model
        arch = self.run.model_architecture
        if arch == "ANN-5-Layer":
            hidden_dims = [128, 64, 32, 16]
        else:
            hidden_dims = [64, 32] # Default ANN-3-Layer
            
        model = TabularANN(input_dim=6, hidden_dims=hidden_dims, output_dim=2).to(self.device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(model.parameters(), lr=self.run.learning_rate)
        
        best_model_wts = model.state_dict()
        best_acc = 0.0
        class_names = ["Not Placed", "Placed"]
        
        for epoch in range(self.run.epochs):
            epoch_metrics = TrainingMetric(training_run_id=self.run_id, epoch=epoch + 1)
            
            # Train phase
            model.train()
            running_loss = 0.0
            running_corrects = 0
            
            for inputs, labels in train_loader:
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                _, preds = torch.max(outputs, 1)
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data).item()
                
            train_loss = running_loss / len(train_ds)
            train_acc = running_corrects / len(train_ds)
            epoch_metrics.train_loss = train_loss
            epoch_metrics.train_accuracy = train_acc
            
            # Val phase
            model.eval()
            running_val_loss = 0.0
            running_val_corrects = 0
            
            with torch.no_grad():
                for inputs, labels in val_loader:
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    
                    _, preds = torch.max(outputs, 1)
                    running_val_loss += loss.item() * inputs.size(0)
                    running_val_corrects += torch.sum(preds == labels.data).item()
                    
            val_loss = running_val_loss / len(val_ds)
            val_acc = running_val_corrects / len(val_ds)
            
            epoch_metrics.val_loss = val_loss
            epoch_metrics.val_accuracy = val_acc
            
            if val_acc > best_acc:
                best_acc = val_acc
                best_model_wts = model.state_dict()
                
            self.db.add(epoch_metrics)
            self.db.commit()
            
        # Save Best Model
        model.load_state_dict(best_model_wts)
        model_dir = os.path.join(MODELS_DIR, f"visionai_run_{self.run_id}")
        os.makedirs(model_dir, exist_ok=True)
        model_path = os.path.join(model_dir, "model_weights.pth")
        torch.save(model.state_dict(), model_path)
        
        # Save Scaler parameters
        scaler_data = {
            "mean": list(cast(Any, scaler).mean_) if cast(Any, scaler).mean_ is not None else [],
            "scale": list(cast(Any, scaler).scale_) if cast(Any, scaler).scale_ is not None else []
        }
        with open(os.path.join(model_dir, "scaler.json"), "w") as f:
            json.dump(scaler_data, f)
            
        # Save classes
        with open(os.path.join(model_dir, "classes.json"), "w") as f:
            json.dump(class_names, f)
            
        # Create Model Info
        model_info = ModelInfo(
            name=f"Placement ANN {arch} Run {self.run_id}",
            version=f"v{self.run_id}",
            architecture=arch,
            dataset_id=self.run.dataset_id,
            file_path=model_path,
            classes_json=json.dumps(class_names),
            accuracy=float(best_acc),
            is_active=True
        )
        self.db.add(model_info)
        self.run.status = "COMPLETED"
        self.db.commit()

    def _train_vision(self, dataset_dir: str):
        # Setup transforms
        data_transforms = {
            'train': transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            'val': transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
        }

        # Load dataset
        full_dataset = datasets.ImageFolder(dataset_dir)
        class_names = full_dataset.classes
        num_classes = len(class_names)
        
        # Split into train/val
        train_size = int(0.8 * len(full_dataset))
        val_size = len(full_dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])
        
        cast(Any, train_dataset.dataset).transform = data_transforms['train']
        cast(Any, val_dataset.dataset).transform = data_transforms['val']
        
        dataloaders = {
            'train': torch.utils.data.DataLoader(train_dataset, batch_size=self.run.batch_size, shuffle=True, num_workers=0),
            'val': torch.utils.data.DataLoader(val_dataset, batch_size=self.run.batch_size, shuffle=False, num_workers=0)
        }
        dataset_sizes = {'train': len(train_dataset), 'val': len(val_dataset)}

        # Setup Model
        model_ft: Any
        optimizer_ft: Any
        if self.run.model_architecture == "EfficientNet-B0":
            model_ft = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
            for param in model_ft.parameters():
                param.requires_grad = False
            num_ftrs = cast(Any, model_ft.classifier[1]).in_features
            model_ft.classifier[1] = nn.Linear(num_ftrs, num_classes)
            optimizer_ft = optim.AdamW(model_ft.classifier.parameters(), lr=self.run.learning_rate)
        elif self.run.model_architecture == "ResNet50":
            model_ft = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
            for param in model_ft.parameters():
                param.requires_grad = False
            num_ftrs = cast(Any, model_ft.fc).in_features
            model_ft.fc = nn.Linear(num_ftrs, num_classes)
            optimizer_ft = optim.AdamW(model_ft.fc.parameters(), lr=self.run.learning_rate)
        elif self.run.model_architecture == "ConvNeXt-Tiny":
            model_ft = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.DEFAULT)
            for param in model_ft.parameters():
                param.requires_grad = False
            num_ftrs = cast(Any, model_ft.classifier[2]).in_features
            model_ft.classifier[2] = nn.Linear(num_ftrs, num_classes)
            optimizer_ft = optim.AdamW(model_ft.classifier.parameters(), lr=self.run.learning_rate)
        elif self.run.model_architecture == "ViT-B/16":
            model_ft = models.vit_b_16(weights=models.ViT_B_16_Weights.DEFAULT)
            for param in model_ft.parameters():
                param.requires_grad = False
            num_ftrs = cast(Any, model_ft.heads.head).in_features
            model_ft.heads.head = nn.Linear(num_ftrs, num_classes)
            optimizer_ft = optim.AdamW(model_ft.heads.parameters(), lr=self.run.learning_rate)
        else: # Default MobileNetV3
            model_ft = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.DEFAULT)
            for param in model_ft.parameters():
                param.requires_grad = False
            num_ftrs = cast(Any, model_ft.classifier[3]).in_features
            model_ft.classifier[3] = nn.Linear(num_ftrs, num_classes)
            optimizer_ft = optim.AdamW(model_ft.classifier.parameters(), lr=self.run.learning_rate)

        model_ft = model_ft.to(self.device)
        criterion = nn.CrossEntropyLoss()
        
        best_model_wts = model_ft.state_dict()
        best_acc = 0.0
        
        for epoch in range(self.run.epochs):
            print(f"Epoch {epoch}/{self.run.epochs - 1}")
            print('-' * 10)
            
            epoch_metrics = TrainingMetric(training_run_id=self.run_id, epoch=epoch + 1)

            for phase in ['train', 'val']:
                if phase == 'train':
                    model_ft.train()
                else:
                    model_ft.eval()
                    
                running_loss = 0.0
                running_corrects: Any = torch.tensor(0, device=self.device)

                for inputs, labels in dataloaders[phase]:
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)

                    optimizer_ft.zero_grad()

                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model_ft(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)

                        if phase == 'train':
                            loss.backward()
                            optimizer_ft.step()

                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)

                epoch_loss = running_loss / dataset_sizes[phase]
                epoch_acc = float((running_corrects.double() / dataset_sizes[phase]).item())

                if phase == 'train':
                    epoch_metrics.train_loss = epoch_loss
                    epoch_metrics.train_accuracy = epoch_acc
                else:
                    epoch_metrics.val_loss = epoch_loss
                    epoch_metrics.val_accuracy = epoch_acc
                    
                    if epoch_acc > best_acc:
                        best_acc = epoch_acc
                        best_model_wts = model_ft.state_dict()
            
            self.db.add(epoch_metrics)
            self.db.commit()
            
        # Save Best Model
        model_ft.load_state_dict(best_model_wts)
        model_dir = os.path.join(MODELS_DIR, f"visionai_run_{self.run_id}")
        os.makedirs(model_dir, exist_ok=True)
        model_path = os.path.join(model_dir, "model_weights.pth")
        torch.save(model_ft.state_dict(), model_path)
        
        # Save classes
        with open(os.path.join(model_dir, "classes.json"), "w") as f:
            json.dump(class_names, f)
            
        # Create Model Info
        model_info = ModelInfo(
            name=f"VisionAI {self.run.model_architecture} Run {self.run_id}",
            version=f"v{self.run_id}",
            architecture=self.run.model_architecture,
            dataset_id=self.run.dataset_id,
            file_path=model_path,
            classes_json=json.dumps(class_names),
            accuracy=float(best_acc),
            is_active=True
        )
        self.db.add(model_info)
        self.run.status = "COMPLETED"
        self.db.commit()

def run_training_task(run_id: int):
    engine = TrainingEngine(run_id)
    engine.start_training()
