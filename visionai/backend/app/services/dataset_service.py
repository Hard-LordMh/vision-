import os
import shutil
import uuid
import zipfile

from PIL import Image
from sqlalchemy.orm import Session

from app.models.schema import Dataset, DatasetClass

STORAGE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "storage")
DATASETS_DIR = os.path.join(STORAGE_DIR, "datasets")
UPLOADS_DIR = os.path.join(STORAGE_DIR, "uploads")

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}

def ensure_dirs():
    os.makedirs(DATASETS_DIR, exist_ok=True)
    os.makedirs(UPLOADS_DIR, exist_ok=True)

def process_dataset_zip(db: Session, zip_path: str, dataset_name: str, source: str = "User Upload") -> Dataset:
    ensure_dirs()
    
    # Create dataset record
    dataset = Dataset(name=dataset_name, source=source)
    db.add(dataset)
    db.commit()
    db.refresh(dataset)
    
    dataset_id_dir = os.path.join(DATASETS_DIR, str(dataset.id))
    raw_dir = os.path.join(dataset_id_dir, "raw")
    processed_dir = os.path.join(dataset_id_dir, "processed")
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(processed_dir, exist_ok=True)
    
    # Extract zip
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(raw_dir)
    
    # Find classes
    # Look for folders containing images. 
    # If the root has folders, those might be classes. If it's train/val/test, we might need to dig deeper.
    # To simplify, we will scan all subdirectories. Any directory that contains images directly will be mapped.
    # But often zip files contain a top-level folder (e.g. `fruit_dataset/`).
    
    class_images_map = {}
    
    for root, dirs, files in os.walk(raw_dir):
        images_in_dir = [f for f in files if os.path.splitext(f)[1].lower() in VALID_EXTENSIONS]
        if images_in_dir:
            class_name = os.path.basename(root)
            # if we have train/val/test split, the class is the leaf, we merge them.
            if class_name in class_images_map:
                class_images_map[class_name].extend([os.path.join(root, img) for img in images_in_dir])
            else:
                class_images_map[class_name] = [os.path.join(root, img) for img in images_in_dir]

    total_valid = 0
    
    for class_name, img_paths in class_images_map.items():
        if len(img_paths) == 0:
            continue
            
        # Add class
        db_class = DatasetClass(dataset_id=dataset.id, name=class_name)
        db.add(db_class)
        db.commit()
        db.refresh(db_class)
        
        class_dir = os.path.join(processed_dir, class_name)
        os.makedirs(class_dir, exist_ok=True)
        
        for img_path in img_paths:
            try:
                # Validate image
                with Image.open(img_path) as img:
                    img.verify() # verify it's a valid image
                
                # Copy to processed folder
                dest_path = os.path.join(class_dir, os.path.basename(img_path))
                # avoid collision
                if os.path.exists(dest_path):
                    dest_path = os.path.join(class_dir, f"{uuid.uuid4().hex}_{os.path.basename(img_path)}")
                shutil.copy2(img_path, dest_path)
                total_valid += 1
            except Exception:
                # Invalid image, ignore
                pass

    dataset.total_images = total_valid
    dataset.class_count = len(class_images_map)
    db.commit()
    db.refresh(dataset)
    
    # cleanup raw
    shutil.rmtree(raw_dir, ignore_errors=True)
    
    return dataset

def get_dataset_processed_dir(dataset_id: int):
    return os.path.join(DATASETS_DIR, str(dataset_id), "processed")
