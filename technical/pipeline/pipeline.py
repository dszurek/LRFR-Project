"""End-to-end low-resolution facial recognition pipeline.

The pipeline stitches together the custom DSR super-resolution network and the
EdgeFace recognition backbone so that a USB camera capture (or any low
resolution input) can be upscaled prior to identification.

The implementation keeps footprint low to match Raspberry Pi 5 (4GB) class
hardware: all operations default to CPU execution, memory copies are avoided,
optional INT8 TorchScript models are supported, and preprocessing is expressed
in PyTorch to leverage vectorised kernels.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
from PIL import Image

try:  # Torch might not be available in the VS Code analysis environment
    import torch
    import torch.nn.functional as F
    from torchvision import transforms
except ModuleNotFoundError:  # pragma: no cover - executed only on dev host
    torch = None  # type: ignore[assignment]
    F = None  # type: ignore[assignment]
    transforms = None  # type: ignore[assignment]

from dsr import DSRColor, DSRConfig, load_dsr_model
from facial_rec.edgeface_weights.edgeface import EdgeFace

TensorLike = Union["torch.Tensor", np.ndarray]
ImageInput = Union[str, Path, Image.Image, TensorLike]


def _require_torch() -> None:
    if torch is None or F is None or transforms is None:
        raise RuntimeError(
            "The pipeline requires PyTorch and torchvision. Install platform-appropriate "
            "wheels before using it."
        )


@dataclass
class PipelineConfig:
    """Runtime configuration for the face pipeline."""

    dsr_weights_path: Path = Path("dsr/dsr.pth")
    edgeface_weights_path: Path = Path("facial_rec/edgeface_weights/edgeface_xxs_q.pt")
    device: Union[str, Any] = "cpu"
    recognition_threshold: float = 0.45
    gallery_normalize: bool = True
    num_threads: Optional[int] = 2
    force_full_precision: bool = True  # disable fp16 for embedded CPUs


class IdentityDatabase:
    """Stores reference embeddings and performs cosine similarity search."""

    def __init__(
        self,
        threshold: float,
        normalize_embeddings: bool = True,
    ) -> None:
        _require_torch()
        self.threshold = threshold
        self.normalize_embeddings = normalize_embeddings
        self._labels: List[str] = []
        self._embeddings: List[Any] = []

    @property
    def size(self) -> int:
        return len(self._labels)

    def add(self, label: str, embedding: Any) -> None:
        if self.normalize_embeddings:
            embedding = F.normalize(embedding, dim=-1)
        self._labels.append(label)
        self._embeddings.append(embedding.cpu())

    def extend(self, items: Iterable[Tuple[str, Any]]) -> None:
        for label, embedding in items:
            self.add(label, embedding)

    def lookup(self, query: Any) -> Optional[Tuple[str, float]]:
        if not self._embeddings:
            return None
        query = query.cpu()
        if self.normalize_embeddings:
            query = F.normalize(query, dim=-1)
        stacked = torch.stack(self._embeddings)
        similarities = torch.mv(stacked, query)
        score, index = torch.max(similarities, dim=0)
        score_float = score.item()
        if math.isfinite(score_float) and score_float >= self.threshold:
            return self._labels[int(index)], score_float
        return None


class FaceRecognitionPipeline:
    """Glue code that runs low-res -> high-res -> recognition inference."""

    def __init__(
        self,
        config: PipelineConfig,
        dsr_model: Optional[Any] = None,
        recognition_model: Optional[Any] = None,
        gallery: Optional[IdentityDatabase] = None,
    ) -> None:
        _require_torch()
        self.config = config
        self.device = torch.device(config.device)
        self.gallery = gallery or IdentityDatabase(
            threshold=config.recognition_threshold,
            normalize_embeddings=config.gallery_normalize,
        )

        if config.num_threads is not None:
            max_threads = os.cpu_count() or config.num_threads
            torch.set_num_threads(max(1, min(config.num_threads, max_threads)))

        self.dsr_model = dsr_model or load_dsr_model(
            config.dsr_weights_path, device=self.device
        )
        if config.force_full_precision:
            self.dsr_model = self.dsr_model.to(dtype=torch.float32)

        self.recognition_model = recognition_model or self._load_edgeface_model(
            config.edgeface_weights_path
        )
        self.recognition_model.to(self.device)
        if config.force_full_precision:
            self.recognition_model = self.recognition_model.to(dtype=torch.float32)
        self.recognition_model.eval()

        self.preprocess = transforms.Compose(
            [
                transforms.Resize((112, 112)),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )

    def _load_edgeface_model(self, weights_path: Path) -> Any:
        resolved = Path(weights_path)
        try:
            model = torch.jit.load(resolved, map_location=self.device)
            model.eval()
            return model
        except (RuntimeError, ValueError):
            pass

        model = EdgeFace(back="edgeface_s")  # default backbone
        state_dict = torch.load(resolved, map_location="cpu")
        if "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
        elif "model_state_dict" in state_dict:
            state_dict = state_dict["model_state_dict"]

        clean_state_dict: Dict[str, Any] = {}
        for key, value in state_dict.items():
            clean_key = key
            if clean_key.startswith("model."):
                clean_key = clean_key[len("model.") :]
            clean_state_dict[clean_key] = value

        missing, unexpected = model.load_state_dict(clean_state_dict, strict=False)
        if missing:
            print(f"[EdgeFace] Missing keys: {missing}")
        if unexpected:
            print(f"[EdgeFace] Unexpected keys: {unexpected}")
        return model

    # ------------------------------------------------------------------ helpers
    def _to_tensor(self, image: ImageInput) -> Any:
        if isinstance(image, torch.Tensor):
            tensor = image
        elif isinstance(image, np.ndarray):
            if image.ndim != 3 or image.shape[2] not in (3, 4):
                raise ValueError("Expected an HxWx3 or HxWx4 numpy array")
            array = image[..., :3]
            array = array.astype("float32") / 255.0
            tensor = torch.from_numpy(array).permute(2, 0, 1)
        else:
            if isinstance(image, (str, Path)):
                pil = Image.open(image).convert("RGB")
            elif isinstance(image, Image.Image):
                pil = image.convert("RGB")
            else:
                raise TypeError(f"Unsupported input type: {type(image)!r}")
            tensor = transforms.functional.to_tensor(pil)
        if tensor.dim() == 3:
            tensor = tensor.unsqueeze(0)
        return tensor.to(self.device)

    @staticmethod
    def _tensor_to_image(tensor: Any) -> Image.Image:
        tensor = tensor.detach().cpu().clamp(0.0, 1.0)
        array = tensor.squeeze(0).permute(1, 2, 0).numpy()
        array = (array * 255.0).astype("uint8")
        return Image.fromarray(array)

    # ---------------------------------------------------------------- interface
    def upscale(self, image: ImageInput) -> Any:
        _require_torch()
        tensor = self._to_tensor(image)
        with torch.inference_mode():
            output = self.dsr_model(tensor)
        return torch.clamp(output, 0.0, 1.0)

    def infer_embedding(self, high_res_tensor: Any) -> Any:
        _require_torch()
        transformed = self.preprocess(high_res_tensor)
        with torch.inference_mode():
            embedding = self.recognition_model(transformed)
        if embedding.ndim == 2:
            embedding = embedding.squeeze(0)
        if embedding.ndim != 1:
            raise RuntimeError("Recognition model returned unexpected shape")
        return F.normalize(embedding, dim=-1)

    def run(self, image: ImageInput) -> Dict[str, object]:
        _require_torch()
        sr = self.upscale(image)
        embedding = self.infer_embedding(sr)
        identity = self.gallery.lookup(embedding)
        result: Dict[str, object] = {
            "identity": identity[0] if identity else None,
            "score": identity[1] if identity else None,
            "embedding": embedding.cpu(),
            "sr_image": self._tensor_to_image(sr),
        }
        return result

    # ------------------------------------------------------ gallery utilities
    def register_identity(
        self,
        label: str,
        image: ImageInput,
        average_over: int = 1,
    ) -> None:
        _require_torch()
        embeddings: List[Any] = []
        for _ in range(max(1, average_over)):
            sr = self.upscale(image)
            embeddings.append(self.infer_embedding(sr))
        stacked = torch.stack(embeddings)
        mean_embedding = torch.mean(stacked, dim=0)
        self.gallery.add(label, mean_embedding)


def build_pipeline(config: Optional[PipelineConfig] = None) -> FaceRecognitionPipeline:
    return FaceRecognitionPipeline(config or PipelineConfig())
