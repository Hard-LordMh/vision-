from datetime import datetime, timezone
from typing import List, Optional

from sqlalchemy import (
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database.database import Base


class Dataset(Base):
    __tablename__ = "datasets"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    name: Mapped[str] = mapped_column(String, index=True)
    source: Mapped[str] = mapped_column(String, default="User Upload")
    license: Mapped[str] = mapped_column(String, default="Unknown")
    total_images: Mapped[int] = mapped_column(Integer, default=0)
    class_count: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(timezone.utc).replace(tzinfo=None))

    classes: Mapped[List["DatasetClass"]] = relationship("DatasetClass", back_populates="dataset", cascade="all, delete-orphan")
    runs: Mapped[List["TrainingRun"]] = relationship("TrainingRun", back_populates="dataset", cascade="all, delete-orphan")
    models: Mapped[List["ModelInfo"]] = relationship("ModelInfo", back_populates="dataset")


class DatasetClass(Base):
    __tablename__ = "dataset_classes"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    dataset_id: Mapped[int] = mapped_column(Integer, ForeignKey("datasets.id"))
    name: Mapped[str] = mapped_column(String)
    
    dataset: Mapped["Dataset"] = relationship("Dataset", back_populates="classes")


class TrainingRun(Base):
    __tablename__ = "training_runs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    dataset_id: Mapped[int] = mapped_column(Integer, ForeignKey("datasets.id"))
    model_name: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    model_architecture: Mapped[str] = mapped_column(String)
    status: Mapped[str] = mapped_column(String, default="READY") # READY, TRAINING, COMPLETED, FAILED
    epochs: Mapped[int] = mapped_column(Integer)
    batch_size: Mapped[int] = mapped_column(Integer)
    learning_rate: Mapped[float] = mapped_column(Float)
    started_at: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(timezone.utc).replace(tzinfo=None))
    completed_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)

    dataset: Mapped["Dataset"] = relationship("Dataset", back_populates="runs")
    metrics: Mapped[List["TrainingMetric"]] = relationship("TrainingMetric", back_populates="run", cascade="all, delete-orphan")


class TrainingMetric(Base):
    __tablename__ = "training_metrics"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    training_run_id: Mapped[int] = mapped_column(Integer, ForeignKey("training_runs.id"))
    epoch: Mapped[int] = mapped_column(Integer)
    train_loss: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    val_loss: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    train_accuracy: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    val_accuracy: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    run: Mapped["TrainingRun"] = relationship("TrainingRun", back_populates="metrics")


class ModelInfo(Base):
    __tablename__ = "models"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    name: Mapped[str] = mapped_column(String)
    version: Mapped[str] = mapped_column(String)
    architecture: Mapped[str] = mapped_column(String)
    dataset_id: Mapped[int] = mapped_column(Integer, ForeignKey("datasets.id"))
    file_path: Mapped[str] = mapped_column(String)
    classes_json: Mapped[str] = mapped_column(Text) # JSON list of class names
    accuracy: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    is_active: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(timezone.utc).replace(tzinfo=None))
    
    dataset: Mapped["Dataset"] = relationship("Dataset", back_populates="models")
    predictions: Mapped[List["Prediction"]] = relationship("Prediction", back_populates="model", cascade="all, delete-orphan")


class Prediction(Base):
    __tablename__ = "predictions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    model_id: Mapped[int] = mapped_column(Integer, ForeignKey("models.id"))
    image_path: Mapped[str] = mapped_column(String)
    predicted_class: Mapped[str] = mapped_column(String)
    confidence: Mapped[float] = mapped_column(Float)
    top_3_json: Mapped[str] = mapped_column(Text) # JSON of Top 3 predictions
    inference_time_ms: Mapped[float] = mapped_column(Float)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(timezone.utc).replace(tzinfo=None))

    model: Mapped["ModelInfo"] = relationship("ModelInfo", back_populates="predictions")

