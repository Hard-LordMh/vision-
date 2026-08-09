from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict


class DatasetClassBase(BaseModel):
    name: str

class DatasetClassResponse(DatasetClassBase):
    id: int
    class_config: dict = {}
    model_config = ConfigDict(from_attributes=True)

class DatasetBase(BaseModel):
    name: str
    source: str = "User Upload"
    license: str = "Unknown"

class DatasetResponse(DatasetBase):
    id: int
    total_images: int
    class_count: int
    created_at: datetime
    classes: list[DatasetClassResponse] = []
    model_config = ConfigDict(from_attributes=True)

class TrainingRunCreate(BaseModel):
    dataset_id: int
    model_architecture: str = "MobileNetV3"
    epochs: int = 10
    batch_size: int = 16
    learning_rate: float = 0.001
    fine_tune: bool = False

class TrainingRunResponse(BaseModel):
    id: int
    dataset_id: int
    model_name: str | None
    model_architecture: str
    status: str
    epochs: int
    batch_size: int
    learning_rate: float
    started_at: datetime
    completed_at: datetime | None
    model_config = ConfigDict(from_attributes=True)

class ModelResponse(BaseModel):
    id: int
    name: str
    version: str
    architecture: str
    dataset_id: int
    accuracy: float | None
    is_active: bool
    created_at: datetime
    classes: list[str]
    model_config = ConfigDict(from_attributes=True)

class PredictionResult(BaseModel):
    predicted_class: str
    confidence: float
    top_3: Any
    inference_time_ms: float
    model_version: str

