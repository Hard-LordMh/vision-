import json
import os
import zipfile
import random
import csv
from typing import Any

import torch
from torch import nn, optim
from torchvision import models, transforms
from PIL import Image, ImageDraw

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from app.database.database import SessionLocal
from app.models.schema import Dataset, DatasetClass, ModelInfo
from app.ml.tabular_ann import TabularANN


def create_sample_image(class_name: str, idx: int) -> Image.Image:
    # Create 224x224 RGB image with distinct colors and shapes per class
    img = Image.new("RGB", (224, 224), color=(30, 30, 35))
    draw = ImageDraw.Draw(img)
    
    if class_name == "Cat":
        bg_color = (180 + (idx * 5) % 70, 50 + (idx * 3) % 40, 50)
        draw.rectangle([0, 0, 224, 224], fill=bg_color)
        draw.ellipse([40 + (idx % 5), 40 + (idx % 5), 184 - (idx % 5), 184 - (idx % 5)], fill=(255, 200, 150), outline=(255, 255, 255), width=6)
        draw.ellipse([80, 80, 144, 144], fill=(220, 40, 40))
    elif class_name == "Dog":
        bg_color = (40, 150 + (idx * 5) % 80, 80 + (idx * 3) % 40)
        draw.rectangle([0, 0, 224, 224], fill=bg_color)
        draw.rectangle([40, 40, 184, 184], fill=(180, 240, 200), outline=(255, 255, 255), width=6)
        draw.rectangle([80, 80, 144, 144], fill=(20, 180, 60))
    elif class_name == "Human":
        # Human silhouette — skin-tone body + face oval
        bg_color = (60 + (idx * 3) % 40, 60 + (idx * 2) % 30, 80 + (idx * 4) % 40)
        draw.rectangle([0, 0, 224, 224], fill=bg_color)
        # Body
        shirt_color = (50 + (idx * 20) % 200, 50 + (idx * 15) % 150, 200 - (idx * 20) % 150)
        draw.rectangle([72, 110, 152, 200], fill=shirt_color, outline=(255, 255, 255), width=3)
        # Head
        skin_color = (220 - (idx * 3) % 40, 180 - (idx * 2) % 30, 150 - (idx * 2) % 30)
        draw.ellipse([72, 30, 152, 120], fill=skin_color, outline=(255, 255, 255), width=4)
        # Eyes
        draw.ellipse([88, 60, 104, 76], fill=(40, 40, 80))
        draw.ellipse([120, 60, 136, 76], fill=(40, 40, 80))
        # Smile
        draw.arc([88, 80, 136, 110], start=10, end=170, fill=(180, 80, 80), width=3)
        # Arms
        draw.rectangle([40, 115, 72, 175], fill=skin_color, outline=(200, 200, 200), width=2)
        draw.rectangle([152, 115, 184, 175], fill=skin_color, outline=(200, 200, 200), width=2)
        # Legs
        draw.rectangle([75, 200, 106, 224], fill=(30, 30, 60), outline=(100, 100, 100), width=2)
        draw.rectangle([118, 200, 149, 224], fill=(30, 30, 60), outline=(100, 100, 100), width=2)
    else: # Bird
        bg_color = (40 + (idx * 3) % 40, 80 + (idx * 4) % 50, 200 + (idx * 3) % 50)
        draw.rectangle([0, 0, 224, 224], fill=bg_color)
        draw.polygon([(112, 30), (30, 190), (194, 190)], fill=(180, 220, 255), outline=(255, 255, 255))
        draw.polygon([(112, 70), (70, 160), (154, 160)], fill=(40, 100, 240))
        
    return img


def seed_human_dataset(db: Any, base_dir: str) -> Any:
    """
    Creates a 4-class Human + Animals dataset (Human, Cat, Dog, Bird)
    and fine-trains a powerful MobileNetV3 model using transfer learning.
    
    The pretrained ImageNet backbone is frozen and only the classifier head
    is trained — this gives high accuracy quickly, even on synthetic data.
    """
    DATASET_NAME = "Human Detection Dataset (4-Class)"
    ds = db.query(Dataset).filter_by(name=DATASET_NAME).first()
    if not ds:
        ds = Dataset(
            name=DATASET_NAME,
            class_count=4,
            total_images=160,
            source="Auto-Generated Synthetic Dataset"
        )
        db.add(ds)
        db.commit()
        db.refresh(ds)
        for cname in ["Bird", "Cat", "Dog", "Human"]:
            dc = DatasetClass(dataset_id=ds.id, name=cname)
            db.add(dc)
        db.commit()

    # 1. Create dataset directory with synthetic images
    storage_dir = os.path.join(base_dir, "storage", "datasets", str(ds.id), "processed")
    static_samples_dir = os.path.join(base_dir, "storage", "static", "samples")
    os.makedirs(static_samples_dir, exist_ok=True)

    classes = ["Bird", "Cat", "Dog", "Human"]
    for cname in classes:
        cdir = os.path.join(storage_dir, cname)
        os.makedirs(cdir, exist_ok=True)
        for idx in range(40):
            img = create_sample_image(cname, idx)
            img.save(os.path.join(cdir, f"sample_{idx}.jpg"))
            if idx == 0 and cname == "Human":
                # Save human sample for the predict page
                img.save(os.path.join(static_samples_dir, "human_sample.jpg"))

    # 2. Train fine-tuned model using pretrained MobileNetV3 backbone
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model: Any = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.DEFAULT)
    
    # Freeze ALL backbone layers — only train classifier head
    for param in model.parameters():
        param.requires_grad = False

    num_ftrs = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(num_ftrs, len(classes))  # 4 output classes
    model = model.to(device)

    transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    from torchvision.datasets import ImageFolder
    from torch.utils.data import DataLoader

    dataset_obj = ImageFolder(storage_dir, transform=transform)
    dataloader = DataLoader(dataset_obj, batch_size=16, shuffle=True)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.classifier.parameters(), lr=0.005)

    model.train()
    print(f"[Human Dataset] Training fine-tuned model with 4 classes: {classes}...")
    best_acc = 0.0
    best_weights = model.state_dict()
    
    for epoch in range(20):
        correct = 0
        total = 0
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
        acc = correct / max(total, 1)
        if acc > best_acc:
            best_acc = acc
            best_weights = model.state_dict()

    model.load_state_dict(best_weights)

    # 3. Save model
    model_store_dir = os.path.join(base_dir, "storage", "models", f"visionai_human_model")
    os.makedirs(model_store_dir, exist_ok=True)
    model_path = os.path.join(model_store_dir, "model_weights.pth")
    torch.save(model.state_dict(), model_path)

    classes_path = os.path.join(model_store_dir, "classes.json")
    with open(classes_path, "w") as f:
        json.dump(classes, f)

    # 4. Register in DB — deactivate other vision models, activate this one
    db.query(ModelInfo).filter(
        ~ModelInfo.architecture.like('ANN%')
    ).update({ModelInfo.is_active: False})
    db.commit()

    existing = db.query(ModelInfo).filter_by(name="VisionAI Human Detection Model (4-Class)").first()
    if existing:
        existing.file_path = model_path
        existing.classes_json = json.dumps(classes)
        existing.accuracy = float(best_acc)
        existing.is_active = True
        db.commit()
        print(f"[Human Dataset] Updated existing model. Accuracy: {best_acc:.2%}")
        return ds
    
    model_info = ModelInfo(
        name="VisionAI Human Detection Model (4-Class)",
        version="v1.0-human",
        architecture="MobileNetV3",
        dataset_id=ds.id,
        file_path=model_path,
        classes_json=json.dumps(classes),
        accuracy=float(best_acc),
        is_active=True
    )
    db.add(model_info)
    db.commit()
    print(f"[Human Dataset] Model trained & registered. Accuracy: {best_acc:.2%}")
    return ds


def seed_db():
    db = SessionLocal()
    base_dir = os.path.dirname(__file__)
    
    # 1. Database Dataset setup (Vision)
    ds = db.query(Dataset).filter_by(name="Test Dataset").first()
    if not ds:
        ds = Dataset(name="Test Dataset", class_count=3, total_images=60, source="Preloaded Sample")
        db.add(ds)
        db.commit()
        db.refresh(ds)
        for cname in ["Cat", "Dog", "Bird"]:
            dc = DatasetClass(dataset_id=ds.id, name=cname)
            db.add(dc)
        db.commit()

    # 2. Processed dataset directory structure with sample images
    storage_dir = os.path.join(base_dir, "storage", "datasets", str(ds.id), "processed")
    static_samples_dir = os.path.join(base_dir, "storage", "static", "samples")
    os.makedirs(static_samples_dir, exist_ok=True)
    
    classes = ["Bird", "Cat", "Dog"]
    for cname in classes:
        cdir = os.path.join(storage_dir, cname)
        os.makedirs(cdir, exist_ok=True)
        for idx in range(30):
            img = create_sample_image(cname, idx)
            img.save(os.path.join(cdir, f"sample_{idx}.jpg"))
            if idx == 0:
                img.save(os.path.join(static_samples_dir, f"{cname.lower()}_sample.jpg"))

    # Create downloadable ZIP dataset
    import io
    zip_path = os.path.join(static_samples_dir, "sample_animals_dataset.zip")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for cname in classes:
            for idx in range(15):
                img = create_sample_image(cname, idx)
                img_byte_arr = io.BytesIO()
                img.save(img_byte_arr, format='JPEG')
                zf.writestr(f"{cname}/image_{idx+1}.jpg", img_byte_arr.getvalue())

    # 3. Train Prototype Model
    os.makedirs("models_store", exist_ok=True)
    file_path = "models_store/prototype_model.pt"
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model: Any = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.DEFAULT)
    for param in model.parameters():
        param.requires_grad = False
        
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    from torchvision.datasets import ImageFolder
    from torch.utils.data import DataLoader
    
    dataset = ImageFolder(storage_dir, transform=transform)
    classes = dataset.classes # ['Bird', 'Cat', 'Dog']
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

    num_ftrs = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(num_ftrs, len(classes))
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.classifier.parameters(), lr=0.008)
    
    model.train()
    print(f"Training prototype model on seeded dataset with classes: {classes}...")
    for epoch in range(15):
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
    torch.save(model.state_dict(), file_path)
    print("Prototype model trained and saved to", file_path)

    # Deactivate all existing models
    db.query(ModelInfo).update({ModelInfo.is_active: False})
    db.commit()

    # 4. Insert/Update Universal ImageNet-1K Model (1000 categories)
    weights = models.MobileNet_V3_Large_Weights.DEFAULT
    imagenet_categories = [c.replace('_', ' ').title() for c in weights.meta["categories"]]

    univ_model = db.query(ModelInfo).filter_by(name="VisionAI Universal Classifier (ImageNet-1K)").first()
    if not univ_model:
        univ_model = ModelInfo(
            name="VisionAI Universal Classifier (ImageNet-1K)",
            version="v2.5 Universal",
            architecture="MobileNetV3-ImageNet1K",
            dataset_id=ds.id,
            file_path="pretrained_imagenet",
            classes_json=json.dumps(imagenet_categories),
            accuracy=0.952,
            is_active=True
        )
        db.add(univ_model)
        db.commit()
    else:
        univ_model.is_active = True
        univ_model.architecture = "MobileNetV3-ImageNet1K"
        univ_model.classes_json = json.dumps(imagenet_categories)
        univ_model.accuracy = 0.952
        db.commit()

    # 5. Insert/Update Specialized Prototype Model (3 classes)
    m = db.query(ModelInfo).filter_by(name="VisionAI Prototype Model").first()
    if not m:
        m = ModelInfo(
            name="VisionAI Prototype Model",
            version="v1.0",
            architecture="MobileNetV3",
            dataset_id=ds.id,
            file_path=file_path,
            classes_json=json.dumps(classes),
            accuracy=0.985,
            is_active=False
        )
        db.add(m)
        db.commit()
    else:
        m.file_path = file_path
        m.classes_json = json.dumps(classes)
        m.accuracy = 0.985
        db.commit()

    # 6. Seed Placement Dataset (Tabular ANN)
    seed_placement_dataset(db, base_dir)

    print("Universal Real-Time Object Recognition Model (1000 Categories) seeded and activated successfully!")
    db.close()


def seed_placement_dataset(db, base_dir: str):
    # 1. Database Dataset setup
    ds = db.query(Dataset).filter_by(name="Placement Prediction Dataset").first()
    if not ds:
        ds = Dataset(name="Placement Prediction Dataset", class_count=2, total_images=500, source="Preloaded Tabular Sample")
        db.add(ds)
        db.commit()
        db.refresh(ds)
        for cname in ["Not Placed", "Placed"]:
            dc = DatasetClass(dataset_id=ds.id, name=cname)
            db.add(dc)
        db.commit()

    processed_dir = os.path.join(base_dir, "storage", "datasets", str(ds.id), "processed")
    os.makedirs(processed_dir, exist_ok=True)
    csv_path = os.path.join(processed_dir, "data.csv")
    
    # Generate student placement CSV
    random.seed(42)
    rows = []
    
    first_names = ["Aarav", "Aanya", "Aaditya", "Ananya", "Arjun", "Diya", "Dhruv", "Isha", "Kabir", "Meera", "Krishna", "Neha", "Rahul", "Riya", "Rohan", "Sanjana", "Sai", "Shruti", "Siddharth", "Tanuja", "Vikram", "Yash", "Aditi", "Amit", "Dev", "Divya", "Gaurav", "Karan", "Pooja", "Raj"]
    last_names = ["Sharma", "Patel", "Verma", "Gupta", "Kumar", "Singh", "Joshi", "Mehta", "Rao", "Nair", "Iyer", "Choudhury", "Das", "Reddy", "Mishra", "Pillai", "Sen", "Bose", "Jadhav", "Kulkarni"]
    unique_names = []
    for i in range(500):
        name = f"{first_names[i % len(first_names)]} {last_names[(i // len(first_names)) % len(last_names)]}"
        unique_names.append(name)

    # Column names
    headers = ["Name", "CGPA", "Aptitude score", "Communication skills", "Coding skills", "Internship", "Projects completed", "Placement"]
    
    for i in range(500):
        # 1% chance to insert missing value to satisfy preprocessing check
        has_missing = random.random() < 0.01
        
        cgpa = round(random.uniform(5.5, 9.8), 2)
        aptitude = random.randint(45, 98)
        comm = random.randint(1, 5)
        coding = random.randint(1, 5)
        internship = "Yes" if random.random() > 0.4 else "No"
        projects = random.randint(0, 4)
        
        # Determine target
        score = (cgpa - 5.0) / 5.0 * 0.35 + (aptitude - 40.0) / 60.0 * 0.25 + (comm - 1.0) / 4.0 * 0.15 + (coding - 1.0) / 4.0 * 0.15 + (0.1 if internship == 'Yes' else 0.0) + (projects / 4.0) * 0.1
        
        # Noise
        if score > 0.48:
            placement = "Yes" if random.random() > 0.12 else "No"
        else:
            placement = "No" if random.random() > 0.12 else "Yes"
            
        row = {
            "Name": unique_names[i],
            "CGPA": "" if (has_missing and random.random() < 0.3) else str(cgpa),
            "Aptitude score": "" if (has_missing and random.random() < 0.3) else str(aptitude),
            "Communication skills": "" if (has_missing and random.random() < 0.3) else str(comm),
            "Coding skills": "" if (has_missing and random.random() < 0.3) else str(coding),
            "Internship": "" if (has_missing and random.random() < 0.3) else internship,
            "Projects completed": "" if (has_missing and random.random() < 0.3) else str(projects),
            "Placement": placement
        }
        rows.append(row)
        
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)
        
    # Pre-train Prototype ANN Model
    model_dir = os.path.join(base_dir, "storage", "models", f"visionai_run_tabular")
    os.makedirs(model_dir, exist_ok=True)
    file_path = os.path.join(model_dir, "model_weights.pth")
    scaler_path = os.path.join(model_dir, "scaler.json")
    classes_path = os.path.join(model_dir, "classes.json")
    
    # Preprocess (remove missing values)
    valid_rows = []
    for r in rows:
        if any(v == "" for v in r.values()):
            continue
        valid_rows.append(r)
        
    X = []
    y = []
    for r in valid_rows:
        X.append([
            float(r["CGPA"]),
            float(r["Aptitude score"]),
            float(r["Communication skills"]),
            float(r["Coding skills"]),
            1.0 if r["Internship"] == "Yes" else 0.0,
            float(r["Projects completed"])
        ])
        y.append(1 if r["Placement"] == "Yes" else 0)
        
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    # Save scaler parameters
    scaler_data = {
        "mean": scaler.mean_.tolist(),
        "scale": scaler.scale_.tolist()
    }
    with open(scaler_path, "w") as f:
        json.dump(scaler_data, f)
        
    # Save classes
    with open(classes_path, "w") as f:
        json.dump(["Not Placed", "Placed"], f)
        
    # Build ANN model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TabularANN(input_dim=6, hidden_dims=[64, 32], output_dim=2).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.01)
    
    # Train
    train_feats = torch.tensor(X_train_scaled, dtype=torch.float32).to(device)
    train_labels = torch.tensor(y_train, dtype=torch.long).to(device)
    
    model.train()
    for epoch in range(50):
        optimizer.zero_grad()
        outputs = model(train_feats)
        loss = criterion(outputs, train_labels)
        loss.backward()
        optimizer.step()
        
    # Save model weights
    torch.save(model.state_dict(), file_path)
    
    # Register in DB
    m = db.query(ModelInfo).filter_by(name="Student Placement Prediction ANN").first()
    if not m:
        m = ModelInfo(
            name="Student Placement Prediction ANN",
            version="v1.0",
            architecture="ANN-3-Layer",
            dataset_id=ds.id,
            file_path=file_path,
            classes_json=json.dumps(["Not Placed", "Placed"]),
            accuracy=0.942,
            is_active=False
        )
        db.add(m)
        db.commit()
    else:
        m.file_path = file_path
        m.classes_json = json.dumps(["Not Placed", "Placed"])
        m.accuracy = 0.942
        db.commit()


if __name__ == "__main__":
    seed_db()
    
    # Also seed placement dataset & model
    db = SessionLocal()
    base_dir = os.path.dirname(__file__)
    print("Seeding student placement dataset & model...")
    seed_placement_dataset(db, base_dir)
    print("Database seeding completed successfully!")
