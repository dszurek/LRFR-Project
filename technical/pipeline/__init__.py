"""Pipeline package for low-resolution face recognition."""

from .pipeline import (
    FaceRecognitionPipeline,
    IdentityDatabase,
    PipelineConfig,
    build_pipeline,
)

__all__ = [
    "FaceRecognitionPipeline",
    "IdentityDatabase",
    "PipelineConfig",
    "build_pipeline",
]
