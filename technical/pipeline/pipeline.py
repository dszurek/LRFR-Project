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
    from torchvision import transforms as transforms
except ImportError:
    torch = None  # type: ignore[assignment]
    F = None  # type: ignore[assignment]
    transforms = None  # type: ignore[assignment]

try:
    from technical.dsr import DSRColor, DSRConfig, load_dsr_model
    from technical.facial_rec.edgeface_weights.edgeface import EdgeFace
except (
    ModuleNotFoundError
):  # pragma: no cover - fallback for direct execution inside technical/
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
    edgeface_weights_path: Path = Path(
        "facial_rec/edgeface_weights/edgeface_finetuned.pth"
    )
    device: Union[str, Any] = "cpu"
    recognition_threshold: float = 0.35  # Lowered from 0.45 based on evaluation results
    gallery_normalize: bool = True
    num_threads: Optional[int] = 2
    force_full_precision: bool = True  # disable fp16 for embedded CPUs
    use_tta: bool = (
        False  # Test-time augmentation (flip horizontal) for better embeddings
    )


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

        base_dir = Path(__file__).resolve().parents[1]
        dsr_path = Path(config.dsr_weights_path)
        if not dsr_path.is_absolute():
            candidate = base_dir / dsr_path
            dsr_path = candidate if candidate.exists() else dsr_path

        edgeface_path = Path(config.edgeface_weights_path)
        if not edgeface_path.is_absolute():
            candidate = base_dir / edgeface_path
            edgeface_path = candidate if candidate.exists() else edgeface_path

        if config.num_threads is not None:
            max_threads = os.cpu_count() or config.num_threads
            torch.set_num_threads(max(1, min(config.num_threads, max_threads)))

        self.dsr_model = dsr_model or load_dsr_model(dsr_path, device=self.device)
        if config.force_full_precision:
            self.dsr_model = self.dsr_model.to(dtype=torch.float32)

        self.recognition_model = recognition_model or self._load_edgeface_model(
            edgeface_path
        )
        self.recognition_model.to(self.device)
        if config.force_full_precision:
            self.recognition_model = self.recognition_model.to(dtype=torch.float32)
        self.recognition_model.eval()

        self.preprocess = transforms.Compose(
            [
                # No resize needed - DSR outputs 112×112 directly for EdgeFace
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

        # Load state dict first to check architecture
        try:
            state_dict = torch.load(resolved, map_location="cpu")
        except Exception as e:
            msg = str(e)
            if (
                "Weights only load failed" in msg
                or "UnpicklingError" in msg
                or "unsupported" in msg
            ):
                print(
                    "[EdgeFace] weights-only load failed; attempting controlled retry with pickles allowed."
                )
                import sys
                import importlib

                try:
                    finetune_mod = importlib.import_module(
                        "technical.facial_rec.finetune_edgeface"
                    )
                    if hasattr(finetune_mod, "FinetuneConfig"):
                        main_module = sys.modules.get("__main__")
                        if main_module and not hasattr(main_module, "FinetuneConfig"):
                            setattr(
                                main_module,
                                "FinetuneConfig",
                                getattr(finetune_mod, "FinetuneConfig"),
                            )
                except Exception as import_err:
                    print(
                        f"[EdgeFace] could not import FinetuneConfig ({import_err}); creating stub."
                    )
                    try:
                        from dataclasses import dataclass

                        main_module = sys.modules.get("__main__")
                        if main_module and not hasattr(main_module, "FinetuneConfig"):

                            @dataclass
                            class FinetuneConfig:
                                """Stub for unpickling fine-tuned checkpoint."""

                                pass

                            setattr(main_module, "FinetuneConfig", FinetuneConfig)
                    except Exception:
                        pass

                state_dict = torch.load(
                    resolved, map_location="cpu", weights_only=False
                )
            else:
                raise

        if "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
        elif "model_state_dict" in state_dict:
            state_dict = state_dict["model_state_dict"]

        # Detect architecture from weight keys
        sample_keys = list(state_dict.keys())[:10]
        if any("stem" in key or "stages" in key for key in sample_keys):
            backbone = "edgeface_xxs"  # ConvNeXt architecture
            print("[EdgeFace] Detected ConvNeXt (edgeface_xxs) architecture")
        elif any("features" in key for key in sample_keys):
            backbone = "edgeface_s"  # LDC architecture
            print("[EdgeFace] Detected LDC (edgeface_s) architecture")
        else:
            # Fallback: check filename
            filename = resolved.stem.lower()
            if "xxs" in filename:
                backbone = "edgeface_xxs"
            else:
                backbone = "edgeface_s"
            print(f"[EdgeFace] Using filename-based detection: {backbone}")

        model = EdgeFace(back=backbone)
                "Weights only load failed" in msg
                or "UnpicklingError" in msg
                or "unsupported" in msg
            ):
                print(
                    "[EdgeFace] weights-only load failed; attempting controlled retry with pickles allowed."
                )
                # The checkpoint was saved from __main__ context (finetune script run directly),
                # so unpickler looks for __main__.FinetuneConfig. We create a stub in __main__
                # to let unpickler resolve the reference, then retry loading with weights_only=False.
                import sys
                import importlib

                try:
                    # Import the real FinetuneConfig from finetune module
                    finetune_mod = importlib.import_module(
                        "technical.facial_rec.finetune_edgeface"
                    )
                    if hasattr(finetune_mod, "FinetuneConfig"):
                        # Inject into __main__ so unpickler can find __main__.FinetuneConfig
                        main_module = sys.modules.get("__main__")
                        if main_module and not hasattr(main_module, "FinetuneConfig"):
                            setattr(
                                main_module,
                                "FinetuneConfig",
                                getattr(finetune_mod, "FinetuneConfig"),
                            )
                except Exception as import_err:
                    # If import fails, create a minimal stub dataclass in __main__
                    print(
                        f"[EdgeFace] could not import FinetuneConfig ({import_err}); creating stub."
                    )
                    try:
                        from dataclasses import dataclass

                        main_module = sys.modules.get("__main__")
                        if main_module and not hasattr(main_module, "FinetuneConfig"):

                            @dataclass
                            class FinetuneConfig:
                                """Stub for unpickling fine-tuned checkpoint."""

                                pass

                            setattr(main_module, "FinetuneConfig", FinetuneConfig)
                    except Exception:
                        pass

                # Warning: weights_only=False can execute arbitrary code from the file.
                state_dict = torch.load(
                    resolved, map_location="cpu", weights_only=False
                )
            else:
                raise
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

    def infer_embedding(
        self, high_res_tensor: Any, use_tta: Optional[bool] = None
    ) -> Any:
        _require_torch()
        use_tta = use_tta if use_tta is not None else self.config.use_tta

        transformed = self.preprocess(high_res_tensor)
        with torch.inference_mode():
            embedding = self.recognition_model(transformed)

            if use_tta:
                # Test-time augmentation: average with horizontally flipped version
                flipped = torch.flip(transformed, dims=[-1])  # Flip width dimension
                embedding_flip = self.recognition_model(flipped)
                # Average embeddings before normalization for better stability
                embedding = (embedding + embedding_flip) / 2.0

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
