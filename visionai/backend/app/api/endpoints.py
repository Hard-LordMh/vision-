import asyncio
import os
import shutil
import csv
import time
import torch
from typing import cast

from fastapi import (
    APIRouter,
    BackgroundTasks,
    Depends,
    File,
    HTTPException,
    UploadFile,
    WebSocket,
)
from sqlalchemy.orm import Session

from app.api.schemas import (
    DatasetResponse,
    ModelResponse,
    PredictionResult,
    TrainingRunCreate,
    TrainingRunResponse,
)
from app.database.database import get_db
from app.ml.inference_engine import InferenceEngine
from app.ml.training_engine import run_training_task
from app.models.schema import (
    Dataset,
    DatasetClass,
    ModelInfo,
    TrainingMetric,
    TrainingRun,
)

router = APIRouter()

# --- Datasets & Preloaded Samples ---

@router.get("/samples/info")
def get_sample_info():
    return {
        "samples": [
            {
                "id": "cat",
                "label": "Sample Feline Matrix",
                "class_name": "Cat",
                "filename": "cat_sample.jpg",
                "url": "/static/samples/cat_sample.jpg"
            },
            {
                "id": "dog",
                "label": "Sample Canine Matrix",
                "class_name": "Dog",
                "filename": "dog_sample.jpg",
                "url": "/static/samples/dog_sample.jpg"
            },
            {
                "id": "bird",
                "label": "Sample Avian Matrix",
                "class_name": "Bird",
                "filename": "bird_sample.jpg",
                "url": "/static/samples/bird_sample.jpg"
            },
            {
                "id": "human",
                "label": "Sample Human Matrix",
                "class_name": "Human",
                "filename": "human_sample.jpg",
                "url": "/static/samples/human_sample.jpg"
            }
        ],
        "dataset_zip_url": "/static/samples/sample_animals_dataset.zip"
    }

from app.services.dataset_service import process_dataset_zip

@router.post("/samples/load-dataset", response_model=DatasetResponse)
def load_preloaded_dataset(db: Session = Depends(get_db)):
    zip_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "storage", "static", "samples", "sample_animals_dataset.zip")
    if not os.path.exists(zip_path):
        raise HTTPException(status_code=404, detail="Preloaded sample dataset file not found on server")
    dataset = process_dataset_zip(db, zip_path, "Preloaded Animals Dataset")
    return dataset

@router.post("/samples/load-placement-dataset", response_model=DatasetResponse)
def load_placement_dataset(db: Session = Depends(get_db)):
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    from seed_model import seed_placement_dataset
    try:
        seed_placement_dataset(db, base_dir)
        dataset = db.query(Dataset).filter(Dataset.name == "Placement Prediction Dataset").first()
        if not dataset:
            raise HTTPException(status_code=500, detail="Failed to retrieve placement dataset after seeding")
        return dataset
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to seed placement dataset: {str(e)}")

@router.post("/samples/load-human-dataset", response_model=DatasetResponse)
def load_human_dataset(db: Session = Depends(get_db)):
    """
    Creates a Human + Animals (4-class) dataset and fine-trains a MobileNetV3 model
    using the powerful pretrained ImageNet backbone as feature extractor.
    The trained model is immediately set as active.
    This may take 1-3 minutes depending on CPU/GPU.
    """
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    from seed_model import seed_human_dataset
    try:
        dataset = seed_human_dataset(db, base_dir)
        if not dataset:
            raise HTTPException(status_code=500, detail="Failed to create human dataset")
        return dataset
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to seed human dataset: {str(e)})")

@router.post("/samples/load-real-placement-dataset", response_model=DatasetResponse)
def load_real_placement_dataset(db: Session = Depends(get_db)):
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    
    # Check if dataset already exists
    ds = db.query(Dataset).filter_by(name="Real Campus Recruitment Dataset").first()
    if ds:
        return ds

    # Download CSV from github raw
    import urllib.request
    import csv
    import io
    import random
    
    url = "https://raw.githubusercontent.com/ShuklaPrashant21/Campus_Recruitment/master/Placement_Data_Full_Class.csv"
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    
    try:
        with urllib.request.urlopen(req) as response:
            csv_content = response.read().decode('utf-8')
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch data from ShuklaPrashant21's repo: {str(e)}")
        
    reader = list(csv.DictReader(io.StringIO(csv_content)))
    
    first_names = ["Aarav", "Aanya", "Aaditya", "Ananya", "Arjun", "Diya", "Dhruv", "Isha", "Kabir", "Meera", "Krishna", "Neha", "Rahul", "Riya", "Rohan", "Sanjana", "Sai", "Shruti", "Siddharth", "Tanuja", "Vikram", "Yash", "Aditi", "Amit", "Dev", "Divya", "Gaurav", "Karan", "Pooja", "Raj"]
    last_names = ["Sharma", "Patel", "Verma", "Gupta", "Kumar", "Singh", "Joshi", "Mehta", "Rao", "Nair", "Iyer", "Choudhury", "Das", "Reddy", "Mishra", "Pillai", "Sen", "Bose", "Jadhav", "Kulkarni"]
    unique_names = []
    for i in range(len(reader)):
        name = f"{first_names[i % len(first_names)]} {last_names[(i // len(first_names)) % len(last_names)]}"
        unique_names.append(name)

    mapped_rows = []
    # To keep the generated features reproducible for seed
    random.seed(42)
    
    for idx, row in enumerate(reader):
        # 1. CGPA: degree_p / 10.0
        try:
            degree_p = float(row.get("degree_p", 65.0))
        except:
            degree_p = 65.0
        cgpa = round(degree_p / 10.0, 2)
        
        # 2. Aptitude score: etest_p
        try:
            etest_p = float(row.get("etest_p", 70.0))
        except:
            etest_p = 70.0
        aptitude = int(etest_p)
        
        # 3. Internship: workex
        workex = row.get("workex", "No")
        internship = "Yes" if workex.lower() == "yes" else "No"
        
        # 4. Communication skills: based on mba_p
        try:
            mba_p = float(row.get("mba_p", 60.0))
        except:
            mba_p = 60.0
        if mba_p >= 70:
            comm = 5
        elif mba_p >= 65:
            comm = 4
        elif mba_p >= 60:
            comm = 3
        elif mba_p >= 55:
            comm = 2
        else:
            comm = 1
            
        # 5. Coding skills: based on degree_t
        degree_t = row.get("degree_t", "Comm&Mgmt")
        if degree_t.lower() == "sci&tech":
            coding = random.choice([4, 5])
        elif degree_t.lower() == "comm&mgmt":
            coding = random.choice([2, 3])
        else:
            coding = random.choice([1, 2])
            
        # 6. Projects completed: based on workex
        if workex.lower() == "yes":
            projects = random.choice([2, 3, 4])
        else:
            projects = random.choice([0, 1, 2])
            
        # 7. Placement: status
        status = row.get("status", "Not Placed")
        placement = "Yes" if status.lower() == "placed" else "No"
        
        mapped_rows.append({
            "Name": unique_names[idx],
            "CGPA": str(cgpa),
            "Aptitude score": str(aptitude),
            "Communication skills": str(comm),
            "Coding skills": str(coding),
            "Internship": internship,
            "Projects completed": str(projects),
            "Placement": placement
        })
        
    if not mapped_rows:
        raise HTTPException(status_code=500, detail="Parsed dataset is empty")

    # Create DB entry
    ds = Dataset(name="Real Campus Recruitment Dataset", class_count=2, total_images=len(mapped_rows), source="Kaggle (Campus Recruitment)")
    db.add(ds)
    db.commit()
    db.refresh(ds)
    
    # Save the mapped CSV file
    datasets_dir = os.path.join(base_dir, "storage", "datasets")
    processed_dir = os.path.join(datasets_dir, str(ds.id), "processed")
    os.makedirs(processed_dir, exist_ok=True)
    csv_path = os.path.join(processed_dir, "data.csv")
    
    headers = ["Name", "CGPA", "Aptitude score", "Communication skills", "Coding skills", "Internship", "Projects completed", "Placement"]
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(mapped_rows)
        
    # Add classes
    for cname in ["Not Placed", "Placed"]:
        dc = DatasetClass(dataset_id=ds.id, name=cname)
        db.add(dc)
    db.commit()
    db.refresh(ds)
    
    return ds
@router.get("/datasets", response_model=list[DatasetResponse])
def get_datasets(db: Session = Depends(get_db)):
    return db.query(Dataset).all()

@router.get("/datasets/{dataset_id}/students")
def get_dataset_students(dataset_id: int, search: str = "", db: Session = Depends(get_db)):
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    ds = db.query(Dataset).filter_by(id=dataset_id).first()
    if not ds:
        raise HTTPException(status_code=404, detail="Dataset not found")
        
    csv_path = os.path.join(base_dir, "storage", "datasets", str(dataset_id), "processed", "data.csv")
    if not os.path.exists(csv_path):
        raise HTTPException(status_code=404, detail="Dataset CSV data file not found")
        
    students = []
    import csv
    
    # Populations metrics for baseline comparison
    p_cgpa, p_apt, p_comm, p_code, p_proj, p_count = 0.0, 0.0, 0.0, 0.0, 0.0, 0
    u_cgpa, u_apt, u_comm, u_code, u_proj, u_count = 0.0, 0.0, 0.0, 0.0, 0.0, 0

    with open(csv_path, mode='r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for idx, row in enumerate(reader):
            name = row.get("Name", f"Student #{idx+1}")
            
            # Parse fields
            try:
                cgpa = float(row.get("CGPA", 0.0))
            except:
                cgpa = 0.0
            try:
                aptitude = int(float(row.get("Aptitude score", 0.0)))
            except:
                aptitude = 0
            try:
                comm = int(float(row.get("Communication skills", 0.0)))
            except:
                comm = 0
            try:
                coding = int(float(row.get("Coding skills", 0.0)))
            except:
                coding = 0
            internship = row.get("Internship", "No")
            try:
                projects = int(float(row.get("Projects completed", 0.0)))
            except:
                projects = 0
            placement = row.get("Placement", "No")
            
            # Update population statistics
            if placement.lower() == "yes":
                p_cgpa += cgpa
                p_apt += aptitude
                p_comm += comm
                p_code += coding
                p_proj += projects
                p_count += 1
            else:
                u_cgpa += cgpa
                u_apt += aptitude
                u_comm += comm
                u_code += coding
                u_proj += projects
                u_count += 1

            # Match search filter
            if search.strip() and search.lower() not in name.lower():
                continue

            students.append({
                "name": name,
                "cgpa": cgpa,
                "aptitude": aptitude,
                "comm": comm,
                "coding": coding,
                "internship": internship,
                "projects": projects,
                "placement": placement
            })
            
    stats = {
        "placed": {
            "cgpa": round(p_cgpa / max(p_count, 1), 2),
            "aptitude": round(p_apt / max(p_count, 1), 1),
            "comm": round(p_comm / max(p_count, 1), 1),
            "coding": round(p_code / max(p_count, 1), 1),
            "projects": round(p_proj / max(p_count, 1), 1),
            "count": p_count
        },
        "unplaced": {
            "cgpa": round(u_cgpa / max(u_count, 1), 2),
            "aptitude": round(u_apt / max(u_count, 1), 1),
            "comm": round(u_comm / max(u_count, 1), 1),
            "coding": round(u_code / max(u_count, 1), 1),
            "projects": round(u_proj / max(u_count, 1), 1),
            "count": u_count
        }
    }
    
    return {
        "students": students,
        "stats": stats
    }

@router.post("/datasets/upload", response_model=DatasetResponse)
async def upload_dataset(file: UploadFile = File(...), name: str = "New Dataset", db: Session = Depends(get_db)):
    if not file.filename or (not file.filename.endswith('.zip') and not file.filename.endswith('.csv')):
        raise HTTPException(status_code=400, detail="Only .zip and .csv files are supported")
        
    uploads_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "storage", "uploads")
    os.makedirs(uploads_dir, exist_ok=True)
    file_path = os.path.join(uploads_dir, file.filename)
    
    with open(file_path, "wb") as f:
        f.write(await file.read())
        
    if file.filename.endswith('.csv'):
        try:
            dataset = Dataset(name=name, source="User Upload")
            db.add(dataset)
            db.commit()
            db.refresh(dataset)
            
            datasets_dir = os.path.join(os.path.dirname(uploads_dir), "datasets")
            dest_dir = os.path.join(datasets_dir, str(dataset.id), "processed")
            os.makedirs(dest_dir, exist_ok=True)
            
            shutil.copy2(file_path, os.path.join(dest_dir, "data.csv"))
            
            classes_detected = ["Not Placed", "Placed"]
            row_count = 0
            with open(file_path, mode='r', encoding='utf-8') as csv_file:
                reader = csv.reader(csv_file)
                header = next(reader, None)
                if header:
                    last_col_idx = len(header) - 1
                    target_vals = set()
                    for row in reader:
                        if row:
                            row_count += 1
                            if len(row) > last_col_idx:
                                target_vals.add(row[last_col_idx])
                    if target_vals:
                        classes_detected = sorted([str(v) for v in target_vals])
            
            dataset.total_images = row_count
            dataset.class_count = len(classes_detected)
            db.commit()
            
            for cname in classes_detected:
                mapped_name = cname
                if cname.lower() == "yes":
                    mapped_name = "Placed"
                elif cname.lower() == "no":
                    mapped_name = "Not Placed"
                db_class = DatasetClass(dataset_id=dataset.id, name=mapped_name)
                db.add(db_class)
            db.commit()
            db.refresh(dataset)
            
            return dataset
        except Exception as e:
            db.rollback()
            raise HTTPException(status_code=500, detail=f"Failed to process CSV file: {str(e)}")
        finally:
            if os.path.exists(file_path):
                try:
                    os.remove(file_path)
                except Exception:
                    pass
    else:
        dataset = process_dataset_zip(db, file_path, name)
        return dataset


# --- Training ---

@router.get("/training", response_model=list[TrainingRunResponse])
def get_training_runs(db: Session = Depends(get_db)):
    return db.query(TrainingRun).order_by(TrainingRun.id.desc()).all()

@router.post("/training/start", response_model=TrainingRunResponse)
def start_training(config: TrainingRunCreate, background_tasks: BackgroundTasks, db: Session = Depends(get_db)):
    run = TrainingRun(
        dataset_id=config.dataset_id,
        model_architecture=config.model_architecture,
        epochs=config.epochs,
        batch_size=config.batch_size,
        learning_rate=config.learning_rate,
        status="READY"
    )
    db.add(run)
    db.commit()
    db.refresh(run)
    
    background_tasks.add_task(run_training_task, cast(int, run.id))
    return run

@router.get("/training/{run_id}", response_model=TrainingRunResponse)
def get_training_run(run_id: int, db: Session = Depends(get_db)):
    run = db.query(TrainingRun).filter(TrainingRun.id == run_id).first()
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")
    return run

@router.get("/training/{run_id}/metrics")
def get_training_metrics(run_id: int, db: Session = Depends(get_db)):
    metrics = db.query(TrainingMetric).filter(TrainingMetric.training_run_id == run_id).order_by(TrainingMetric.epoch).all()
    return metrics

@router.websocket("/ws/training/{run_id}")
async def websocket_training(websocket: WebSocket, run_id: int, db: Session = Depends(get_db)):
    await websocket.accept()
    try:
        while True:
            db.expire_all()
            run = db.query(TrainingRun).filter(TrainingRun.id == run_id).first()
            if not run:
                await websocket.close()
                break
            assert run is not None
                
            metrics = db.query(TrainingMetric).filter(TrainingMetric.training_run_id == run_id).order_by(TrainingMetric.epoch.desc()).first()
            
            data = {
                "status": run.status,
                "epoch": metrics.epoch if metrics else 0,
                "epochs": run.epochs,
                "train_loss": metrics.train_loss if metrics else None,
                "val_loss": metrics.val_loss if metrics else None,
                "train_accuracy": metrics.train_accuracy if metrics else None,
                "val_accuracy": metrics.val_accuracy if metrics else None,
            }
            await websocket.send_json(data)
            
            if run.status in ["COMPLETED", "FAILED"]:
                break
                
            await asyncio.sleep(2) # poll every 2 seconds
    except Exception as e:
        print(f"WebSocket Error: {e}")
    finally:
        db.close()

# --- Models ---

@router.get("/models", response_model=list[ModelResponse])
def get_models(db: Session = Depends(get_db)):
    models = db.query(ModelInfo).order_by(ModelInfo.id.desc()).all()
    for m in models:
        m.classes = json.loads(str(m.classes_json))
    return models
    
import json


@router.post("/models/{model_id}/activate")
def activate_model(model_id: int, db: Session = Depends(get_db)):
    db.query(ModelInfo).update({ModelInfo.is_active: False})
    model = db.query(ModelInfo).filter(ModelInfo.id == model_id).first()
    if model:
        model.is_active = True
        db.commit()
    return {"status": "success"}

@router.post("/predict", response_model=PredictionResult)
async def predict_image(file: UploadFile = File(...), threshold: float = 0.60, db: Session = Depends(get_db)):
    # Vision models: NOT ANN architecture, prefer active, fall back to latest
    active_model = (
        db.query(ModelInfo)
        .filter(ModelInfo.is_active == True, ~ModelInfo.architecture.like('ANN%'))
        .order_by(ModelInfo.id.desc())
        .first()
    )
    if not active_model:
        active_model = (
            db.query(ModelInfo)
            .filter(~ModelInfo.architecture.like('ANN%'))
            .order_by(ModelInfo.id.desc())
            .first()
        )
    if not active_model:
        raise HTTPException(status_code=400, detail="No active vision model found. Please train or activate a vision model.")
    assert active_model is not None
        
    uploads_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "storage", "uploads")
    os.makedirs(uploads_dir, exist_ok=True)
    file_path = os.path.join(uploads_dir, f"temp_{file.filename}")
    
    with open(file_path, "wb") as f:
        f.write(await file.read())
        
    engine = InferenceEngine(cast(int, active_model.id))
    result = engine.predict(file_path, threshold)
    
    if os.path.exists(file_path):
        try:
            os.remove(file_path)
        except Exception:
            pass
    
    if "error" in result:
        raise HTTPException(status_code=500, detail=result["error"])
        
    return result


import base64
import io
import time
import zipfile
from PIL import Image

@router.post("/predict/zip")
async def predict_zip(file: UploadFile = File(...), threshold: float = 0.60, db: Session = Depends(get_db)):
    if not file.filename or not file.filename.endswith('.zip'):
        raise HTTPException(status_code=400, detail="Only .zip files are supported")

    # Vision models: NOT ANN architecture, prefer active, fall back to latest
    active_model = (
        db.query(ModelInfo)
        .filter(ModelInfo.is_active == True, ~ModelInfo.architecture.like('ANN%'))
        .order_by(ModelInfo.id.desc())
        .first()
    )
    if not active_model:
        active_model = (
            db.query(ModelInfo)
            .filter(~ModelInfo.architecture.like('ANN%'))
            .order_by(ModelInfo.id.desc())
            .first()
        )
    if not active_model:
        raise HTTPException(status_code=400, detail="No active vision model found. Please train or activate a vision model.")
    assert active_model is not None
        
    contents = await file.read()
    uploads_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "storage", "uploads")
    os.makedirs(uploads_dir, exist_ok=True)
    
    engine = InferenceEngine(cast(int, active_model.id))
    
    predictions = []
    class_counts = {}
    total_start = time.time()
    
    valid_extensions = ('.jpg', '.jpeg', '.png', '.webp', '.bmp', '.JPG', '.JPEG', '.PNG')
    
    with zipfile.ZipFile(io.BytesIO(contents)) as zf:
        namelist = [name for name in zf.namelist() if not name.startswith('__MACOSX') and not name.endswith('/') and name.endswith(valid_extensions)]
        
        for fname in namelist:
            try:
                img_bytes = zf.read(fname)
                temp_filename = f"batch_temp_{os.path.basename(fname)}"
                temp_img_path = os.path.join(uploads_dir, temp_filename)
                with open(temp_img_path, "wb") as f:
                    f.write(img_bytes)
                    
                res = engine.predict(temp_img_path, threshold)
                
                # Thumbnail base64
                data_url = None
                try:
                    with Image.open(temp_img_path) as img:
                        img_copy = img.convert('RGB')
                        img_copy.thumbnail((256, 256))
                        buf = io.BytesIO()
                        img_copy.save(buf, format='JPEG')
                        data_url = "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode('utf-8')
                except Exception as e:
                    print(f"Thumbnail error: {e}")
                    
                if os.path.exists(temp_img_path):
                    try:
                        os.remove(temp_img_path)
                    except Exception:
                        pass
                        
                if "error" not in res:
                    pclass = res["predicted_class"]
                    class_counts[pclass] = class_counts.get(pclass, 0) + 1
                    predictions.append({
                        "filename": fname,
                        "predicted_class": pclass,
                        "confidence": res["confidence"],
                        "top_3": res["top_3"],
                        "inference_time_ms": res["inference_time_ms"],
                        "image_data_url": data_url
                    })
            except Exception as e:
                print(f"Error processing {fname}: {e}")
                
    total_time_ms = (time.time() - total_start) * 1000
    
    return {
        "filename": file.filename or "",
        "total_images": len(predictions),
        "total_inference_time_ms": total_time_ms,
        "class_summary": class_counts,
        "model_version": active_model.version,
        "predictions": predictions
    }


@router.post("/predict/tabular")
def predict_tabular(payload: dict, db: Session = Depends(get_db)):
    # Tabular ANN model: architecture starts with 'ANN', prefer active, fall back to latest
    active_model = (
        db.query(ModelInfo)
        .filter(ModelInfo.is_active == True, ModelInfo.architecture.like('ANN%'))
        .order_by(ModelInfo.id.desc())
        .first()
    )
    if not active_model:
        active_model = (
            db.query(ModelInfo)
            .filter(ModelInfo.architecture.like('ANN%'))
            .order_by(ModelInfo.id.desc())
            .first()
        )
    if not active_model:
        raise HTTPException(status_code=400, detail="No active ANN placement model found. Please train or activate an ANN model.")
        
    arch = str(active_model.architecture)
    # Read inputs
    try:
        cgpa = float(payload.get("cgpa") or 0.0)
        aptitude = float(payload.get("aptitude_score") or payload.get("aptitude") or 0.0)
        comm = float(payload.get("communication_skills") or payload.get("communication") or 0.0)
        coding = float(payload.get("coding_skills") or payload.get("coding") or 0.0)
        internship = str(payload.get("internship") or "No")
        projects = float(payload.get("projects_completed") or payload.get("projects") or 0.0)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid payload formats: {str(e)}")
    
    internship_encoded = 1.0 if internship.lower() == "yes" else 0.0
    features = [cgpa, aptitude, comm, coding, internship_encoded, projects]
    
    # Load model and scaler
    model_dir = os.path.dirname(str(active_model.file_path))
    scaler_path = os.path.join(model_dir, "scaler.json")
    
    if not os.path.exists(scaler_path):
        mean = [7.5, 70.0, 3.0, 3.0, 0.4, 2.0]
        scale = [1.2, 15.0, 1.2, 1.2, 0.5, 1.2]
    else:
        import json
        with open(scaler_path, "r") as f:
            scaler_data = json.load(f)
            mean = scaler_data["mean"]
            scale = scaler_data["scale"]
            
    # Normalize features
    features_scaled = []
    for i in range(len(features)):
        val = (features[i] - mean[i]) / (scale[i] if scale[i] != 0 else 1.0)
        features_scaled.append(val)
        
    # Setup PyTorch model
    from app.ml.tabular_ann import TabularANN
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if arch == "ANN-5-Layer":
        hidden_dims = [128, 64, 32, 16]
    else:
        hidden_dims = [64, 32]
        
    model = TabularANN(input_dim=6, hidden_dims=hidden_dims, output_dim=2).to(device)
    
    file_path = str(active_model.file_path)
    if os.path.exists(file_path):
        model.load_state_dict(torch.load(file_path, map_location=device, weights_only=True))
    model.eval()
    
    start_time = time.time()
    with torch.no_grad():
        in_tensor = torch.tensor([features_scaled], dtype=torch.float32).to(device)
        output = model(in_tensor)
        probabilities = torch.nn.functional.softmax(output[0], dim=0)
        
    prob_placed = float(probabilities[1].item())
    prob_not_placed = float(probabilities[0].item())
    
    prediction = "Placed" if prob_placed >= 0.5 else "Not Placed"
    confidence = max(prob_placed, prob_not_placed)
    inference_time_ms = (time.time() - start_time) * 1000
    
    return {
        "prediction": prediction,
        "predicted_class": prediction,
        "placement_probability": prob_placed,
        "confidence": confidence,
        "inference_time_ms": inference_time_ms,
        "model_version": str(active_model.version),
        "top_3": [
            {"class": "Placed", "confidence": prob_placed},
            {"class": "Not Placed", "confidence": prob_not_placed}
        ]
    }


import platform
import os
import torch

@router.get("/system/info")
def get_system_info():
    cuda_available = torch.cuda.is_available()
    device_name = "CPU"
    gpu_memory = None
    
    if cuda_available:
        try:
            device_name = torch.cuda.get_device_name(0)
            properties = torch.cuda.get_device_properties(0)
            gpu_memory = f"{properties.total_memory / (1024 ** 3):.2f} GB"
        except Exception:
            device_name = "GPU (CUDA Available)"
            
    return {
        "pytorch_version": torch.__version__,
        "cuda_available": cuda_available,
        "device_name": device_name,
        "gpu_memory": gpu_memory,
        "cpu_count": os.cpu_count() or 1,
        "os_name": platform.system(),
        "os_release": platform.release(),
    }

