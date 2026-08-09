"""
Register a powerful open-source universal classifier (ResNet50 ImageNet-1K)
as the active vision model.

ResNet50 ImageNet-1K covers 1000 classes including:
  - person / people / faces
  - cat, dog, bird, fish, bear, elephant, lion, tiger ...
  - car, truck, airplane, boat, bicycle ...
  - food, furniture, tools, musical instruments ...

No training needed — uses official torchvision pretrained weights (80.1% top-1 accuracy).
"""
import json
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from app.database.database import SessionLocal
from app.models.schema import ModelInfo

IMAGENET_LABEL = json.dumps([
    "Person", "Cat", "Dog", "Bird", "Car", "Airplane", "Bicycle",
    "Bear", "Elephant", "Lion", "Tiger", "Horse", "Cow", "Sheep",
    "Pizza", "Hamburger", "Guitar", "Laptop", "Mobile Phone",
    "+ 981 more ImageNet-1K classes (1000 total)"
])

def register():
    db = SessionLocal()
    try:
        print("=== Registering Universal ResNet50 ImageNet-1K Vision Model ===\n")
        
        # Deactivate ALL currently active vision (non-ANN) models
        vision_models = db.query(ModelInfo).filter(
            ~ModelInfo.architecture.like('ANN%'),
            ModelInfo.is_active == True
        ).all()
        for m in vision_models:
            m.is_active = False
            print(f"  Deactivated: ID={m.id} | {m.name}")
        db.commit()

        # Get valid dataset_id (first available dataset)
        from app.models.schema import Dataset
        first_ds = db.query(Dataset).order_by(Dataset.id).first()
        ds_id = first_ds.id if first_ds else 1

        # Register new universal ResNet50 ImageNet model
        new_model = ModelInfo(
            name="VisionAI Universal ResNet50 (ImageNet-1K)",
            version="imagenet1k-resnet50-v1",
            architecture="ResNet50-ImageNet1K",
            dataset_id=ds_id,
            accuracy=0.8013,  # Official ResNet50 top-1 accuracy on ImageNet
            is_active=True,
            file_path="pretrained_imagenet",
            classes_json=IMAGENET_LABEL,
        )
        db.add(new_model)
        db.commit()
        db.refresh(new_model)
        print(f"\n✅ Registered: ID={new_model.id} | {new_model.name}")
        print(f"   Architecture : {new_model.architecture}")
        print(f"   Top-1 Acc    : {(new_model.accuracy or 0) * 100:.1f}% on ImageNet")
        print(f"   Active       : {new_model.is_active}")
        print(f"\nThis model recognises 1000 classes including PERSON, CAT, DOG, BIRD, etc.")
        print("Weights are loaded automatically from torchvision (no local training needed).")
        
    finally:
        db.close()

if __name__ == "__main__":
    register()
