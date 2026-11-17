"""Medical image ingestion for DICOM, NIfTI, and ordinary raster images.

The rest of the pipeline consumes PIL images and file paths. This module keeps
that contract unchanged while adding file-level support for single DICOM files
and NIfTI volumes:

* DICOM files are decoded with ``pydicom``, rescaled, windowed, and rendered to a
  cached 8-bit PNG.
* NIfTI volumes are decoded with ``nibabel``, reduced to a representative slice,
  intensity-normalized, and rendered to a cached 8-bit PNG.
* The cached PNG is what the VLM and the IQA-PyTorch tools see. The original
  medical file is never passed to a model or metric that cannot read it.

Only a small allow-list of non-identifying metadata is extracted. Patient and
institution identifiers are intentionally ignored. This is a safeguard, not a
HIPAA/GDPR compliance guarantee: burned-in PHI in pixels still requires a
separate redaction workflow.
"""

from __future__ import annotations

import hashlib
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
from PIL import Image

from src.agentic.privacy import redact_path

NATURAL_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}
DICOM_EXTENSIONS = {".dcm", ".dicom", ".ima"}
NIFTI_EXTENSIONS = {".nii", ".nii.gz"}
SUPPORTED_IMAGE_EXTENSIONS = NATURAL_IMAGE_EXTENSIONS | DICOM_EXTENSIONS | NIFTI_EXTENSIONS

# DICOM modality codes mapped to Medical AgenticIQA profile keys.
_DICOM_MODALITY_TO_PROFILE = {
    "CT": "ct",
    "MR": "mri",
    "US": "ultrasound",
    "MG": "mammography",
    "CR": "radiography",
    "DX": "radiography",
    "XA": "radiography",
    "RF": "radiography",
    "DXA": "radiography",
    "PX": "radiography",
    "OCT": "fundus_oct",
    "OP": "fundus_oct",
    "SM": "histopathology",
}

_BODY_PART_TO_PROFILE = {
    "BREAST": "mammography",
    "SKIN": "dermatology",
    "EXTREMITY": "radiography",
    "CHEST": "radiography",
    "ABDOMEN": "ct",
    "HEAD": "ct",
}


class MedicalImageError(ValueError):
    """Raised when a medical image cannot be decoded or rendered."""


def _lower_name(path: Path) -> str:
    return path.name.lower()


def is_nifti(image_path: str) -> bool:
    """Return True when the path looks like a NIfTI volume."""
    name = _lower_name(Path(image_path))
    return any(name.endswith(ext) for ext in NIFTI_EXTENSIONS)


def is_dicom(image_path: str) -> bool:
    """Return True when the path looks like a DICOM file."""
    name = _lower_name(Path(image_path))
    return any(name.endswith(ext) for ext in DICOM_EXTENSIONS)


def is_medical_image(image_path: str) -> bool:
    """Return True when the path requires medical decoding."""
    return is_dicom(image_path) or is_nifti(image_path)


def _cache_dir() -> Path:
    configured = os.environ.get("AGENTIC_RENDER_CACHE")
    if not configured:
        try:
            from src.utils.config import load_config

            configured = (
                load_config("configs/pipeline.yaml")
                .get("medical", {})
                .get("render_cache_dir")
            )
        except Exception:
            configured = None
    if configured:
        return Path(configured)
    root = Path(os.environ.get("AGENTIC_ROOT", Path.cwd()))
    return root / "data" / "processed" / "render_cache"


def _source_fingerprint(image_path: Path) -> str:
    stat = image_path.stat()
    key = f"{image_path.resolve()}|{stat.st_mtime_ns}|{stat.st_size}|v1"
    return hashlib.sha256(key.encode("utf-8")).hexdigest()[:24]


def _first_value(value: Any) -> Any:
    """Return the first item of a pydicom multi-value, otherwise the value."""
    if value is None:
        return None
    if isinstance(value, (str, bytes)):
        return value
    if isinstance(value, (list, tuple)):
        return value[0] if value else None
    try:
        from pydicom.multival import MultiValue

        if isinstance(value, MultiValue):
            return value[0] if len(value) else None
    except Exception:
        pass
    return value


def _as_float(value: Any, default: float) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _normalize_to_uint8(array: np.ndarray, percentiles: tuple = (1.0, 99.0)) -> np.ndarray:
    """Normalize an array to uint8 using robust percentiles."""
    finite = np.asarray(array, dtype=np.float32)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return np.zeros(array.shape, dtype=np.uint8)

    low, high = np.percentile(finite, percentiles)
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        low = float(finite.min())
        high = float(finite.max())
    if high <= low:
        return np.zeros(array.shape, dtype=np.uint8)

    scaled = (np.asarray(array, dtype=np.float32) - low) / (high - low)
    scaled = np.clip(scaled, 0.0, 1.0)
    return (scaled * 255.0 + 0.5).astype(np.uint8)


def _apply_dicom_window(
    array: np.ndarray,
    center: Optional[float],
    width: Optional[float],
) -> np.ndarray:
    """Apply a DICOM linear VOI window, or robust percentile normalization."""
    if center is not None and width is not None and width > 1.0:
        lower = center - (width - 1.0) / 2.0
        upper = center + (width - 1.0) / 2.0
        if upper > lower:
            scaled = (np.asarray(array, dtype=np.float32) - lower) / (upper - lower)
            scaled = np.clip(scaled, 0.0, 1.0)
            return (scaled * 255.0 + 0.5).astype(np.uint8)

    return _normalize_to_uint8(array)


def _load_dicom_image(image_path: Path) -> Image.Image:
    try:
        import pydicom
        from pydicom.pixel_data_handlers.util import (
            apply_color_lut,
            convert_color_space,
        )
    except ImportError as exc:  # pragma: no cover - depends on optional package
        raise MedicalImageError(
            "DICOM support requires the 'pydicom' package. "
            "Install it with: pip install pydicom"
        ) from exc

    try:
        dataset = pydicom.dcmread(str(image_path), force=True)
        array = dataset.pixel_array
    except Exception as exc:
        raise MedicalImageError(
            f"Unable to decode DICOM pixel data for {redact_path(str(image_path))}. "
            "The file may use a compressed transfer syntax that needs pylibjpeg "
            "plugins, or it may not contain pixel data."
        ) from exc

    samples_per_pixel = int(getattr(dataset, "SamplesPerPixel", 1) or 1)
    photometric = str(getattr(dataset, "PhotometricInterpretation", "") or "").upper()

    # Multi-frame: choose the middle frame as a deterministic representative.
    if array.ndim == 4:
        array = array[array.shape[0] // 2]
    elif array.ndim == 3 and samples_per_pixel <= 1:
        array = array[array.shape[0] // 2]

    if array.ndim == 2:
        array = np.asarray(array, dtype=np.float32)

        slope = _as_float(getattr(dataset, "RescaleSlope", 1.0), 1.0)
        intercept = _as_float(getattr(dataset, "RescaleIntercept", 0.0), 0.0)
        if slope != 1.0 or intercept != 0.0:
            array = array * slope + intercept

        center = _as_float(_first_value(getattr(dataset, "WindowCenter", None)), None)  # type: ignore[arg-type]
        width = _as_float(_first_value(getattr(dataset, "WindowWidth", None)), None)  # type: ignore[arg-type]

        rendered = _apply_dicom_window(array, center, width)
        if photometric == "MONOCHROME1":
            rendered = 255 - rendered

        rgb = np.stack([rendered, rendered, rendered], axis=-1)
        return Image.fromarray(rgb, mode="RGB")

    # Color or palette data.
    if photometric == "PALETTE COLOR":
        try:
            array = apply_color_lut(array, dataset)
        except Exception:
            pass

    if samples_per_pixel > 1 and photometric in {
        "YBR_FULL",
        "YBR_FULL_422",
        "YBR_PARTIAL_422",
        "YBR_PARTIAL_420",
        "YBR_ICT",
        "YBR_RCT",
    }:
        try:
            array = convert_color_space(array, photometric, "RGB")
        except Exception:
            pass

    if array.ndim == 2:
        rgb = np.stack([array, array, array], axis=-1)
    elif array.shape[-1] >= 3:
        rgb = array[..., :3]
    else:
        rgb = np.repeat(array[..., :1], 3, axis=-1)

    return Image.fromarray(_normalize_to_uint8(rgb), mode="RGB")


def _load_nifti_image(image_path: Path) -> Image.Image:
    try:
        import nibabel as nib
    except ImportError as exc:  # pragma: no cover - depends on optional package
        raise MedicalImageError(
            "NIfTI support requires the 'nibabel' package. "
            "Install it with: pip install nibabel"
        ) from exc

    try:
        volume = nib.load(str(image_path))
        data = np.asanyarray(volume.dataobj)
    except Exception as exc:
        raise MedicalImageError(
            f"Unable to decode NIfTI volume {redact_path(str(image_path))}."
        ) from exc

    if data.ndim == 4:
        data = data[..., data.shape[3] // 2]
    if data.ndim == 3:
        data = data[:, :, data.shape[2] // 2]
    if data.ndim > 3:
        raise MedicalImageError(
            "NIfTI volume has more than four dimensions; choose a single "
            "volume or extract a slice before assessment."
        )

    rendered = _normalize_to_uint8(data)
    if rendered.ndim == 2:
        rgb = np.stack([rendered, rendered, rendered], axis=-1)
    else:
        rgb = rendered
    return Image.fromarray(rgb, mode="RGB")


def render_medical_image(image_path: str, force: bool = False) -> Path:
    """Render a DICOM/NIfTI file to a cached PNG and return its path."""
    path = Path(image_path)
    if not path.exists():
        raise ValueError(f"Image file not found: {redact_path(str(path))}")
    if not is_medical_image(str(path)):
        raise ValueError("render_medical_image expects a DICOM or NIfTI file")

    cache_path = _cache_dir() / f"{_source_fingerprint(path)}.png"
    if cache_path.exists() and not force:
        return cache_path

    cache_path.parent.mkdir(parents=True, exist_ok=True)

    if is_dicom(str(path)):
        image = _load_dicom_image(path)
    else:
        image = _load_nifti_image(path)

    fd, temp_name = tempfile.mkstemp(
        prefix="render_", suffix=".png", dir=str(cache_path.parent)
    )
    os.close(fd)
    temp_path = Path(temp_name)
    try:
        image.save(temp_path, format="PNG")
        temp_path.replace(cache_path)
    finally:
        if temp_path.exists():
            temp_path.unlink(missing_ok=True)

    return cache_path


def get_display_image(image_path: str, max_size: Optional[int] = None) -> Image.Image:
    """Load any supported image as a display-ready PIL image.

    Medical files are rendered to a cached PNG first. Raster images are opened
    directly. The returned image is RGB or RGBA and optionally resized so its
    longest edge does not exceed ``max_size``.
    """
    path = Path(image_path)
    if not path.exists():
        raise ValueError(f"Image file not found: {redact_path(str(path))}")

    name = _lower_name(path)
    supported = any(name.endswith(ext) for ext in SUPPORTED_IMAGE_EXTENSIONS)
    if not supported:
        raise ValueError(
            f"Unsupported image format: {path.suffix}. "
            f"Supported formats: {', '.join(sorted(SUPPORTED_IMAGE_EXTENSIONS))}"
        )

    source_path = render_medical_image(str(path)) if is_medical_image(str(path)) else path

    try:
        with Image.open(source_path) as opened:
            opened.load()
            image = opened.copy()
    except Exception as exc:
        raise ValueError(
            f"Failed to load image {redact_path(str(path))}: {exc}"
        ) from exc

    if image.mode not in ("RGB", "RGBA"):
        image = image.convert("RGB")

    if max_size and max(image.size) > max_size:
        original_size = image.size
        ratio = max_size / max(image.size)
        new_size = tuple(max(1, int(dim * ratio)) for dim in image.size)
        image = image.resize(new_size, Image.Resampling.LANCZOS)
        print(
            f"Warning: Resized image from {original_size} to {new_size} "
            f"(max_size={max_size})"
        )

    return image


def get_display_path(image_path: str) -> str:
    """Return a path that existing image tools can read.

    DICOM/NIfTI inputs are rendered to a cached PNG; raster inputs are returned
    unchanged.
    """
    if is_medical_image(image_path):
        return str(render_medical_image(image_path))
    return image_path


def _extract_dicom_metadata(path: Path) -> Dict[str, Any]:
    try:
        import pydicom
    except ImportError as exc:  # pragma: no cover - depends on optional package
        raise MedicalImageError(
            "DICOM support requires the 'pydicom' package. "
            "Install it with: pip install pydicom"
        ) from exc

    try:
        dataset = pydicom.dcmread(str(path), stop_before_pixels=True, force=True)
    except Exception as exc:
        raise MedicalImageError(
            f"Unable to read DICOM metadata for {redact_path(str(path))}."
        ) from exc

    modality_code = str(getattr(dataset, "Modality", "") or "").strip().upper()
    body_part = str(getattr(dataset, "BodyPartExamined", "") or "").strip().upper()
    frames_raw = _first_value(getattr(dataset, "NumberOfFrames", None))
    try:
        frames = int(frames_raw) if frames_raw is not None else 1
    except (TypeError, ValueError):
        frames = 1

    metadata: Dict[str, Any] = {
        "source_format": "dicom",
        "is_medical": True,
        "modality_code": modality_code or None,
        "body_part": body_part or None,
        "rows": int(getattr(dataset, "Rows", 0) or 0) or None,
        "columns": int(getattr(dataset, "Columns", 0) or 0) or None,
        "frames": frames,
        "photometric_interpretation": str(
            getattr(dataset, "PhotometricInterpretation", "") or ""
        )
        or None,
        "bits_allocated": int(getattr(dataset, "BitsAllocated", 0) or 0) or None,
        "slice_thickness_mm": _as_float(
            _first_value(getattr(dataset, "SliceThickness", None)), None  # type: ignore[arg-type]
        ),
        "window_center": _as_float(
            _first_value(getattr(dataset, "WindowCenter", None)), None  # type: ignore[arg-type]
        ),
        "window_width": _as_float(
            _first_value(getattr(dataset, "WindowWidth", None)), None  # type: ignore[arg-type]
        ),
        "modality_hint": infer_modality_hint(
            modality_code=modality_code, body_part=body_part
        ),
    }
    return metadata


def _extract_nifti_metadata(path: Path) -> Dict[str, Any]:
    try:
        import nibabel as nib
    except ImportError as exc:  # pragma: no cover - depends on optional package
        raise MedicalImageError(
            "NIfTI support requires the 'nibabel' package. "
            "Install it with: pip install nibabel"
        ) from exc

    try:
        volume = nib.load(str(path))
        shape = tuple(int(item) for item in volume.shape)
        zooms = tuple(round(float(item), 4) for item in volume.header.get_zooms())
        dtype = str(volume.get_data_dtype())
    except Exception as exc:
        raise MedicalImageError(
            f"Unable to read NIfTI metadata for {redact_path(str(path))}."
        ) from exc

    return {
        "source_format": "nifti",
        "is_medical": True,
        "shape": list(shape),
        "voxel_size_mm": list(zooms),
        "data_type": dtype,
        "modality_hint": None,
        "note": "A single representative slice is rendered for assessment.",
    }


def get_medical_metadata(image_path: str) -> Dict[str, Any]:
    """Return non-identifying metadata for a supported image.

    Only an allow-list of technical attributes is reported. Patient, study, and
    institution identifiers are never read into the result.
    """
    path = Path(image_path)
    if not path.exists():
        raise ValueError(f"Image file not found: {redact_path(str(path))}")

    if is_dicom(str(path)):
        return _extract_dicom_metadata(path)
    if is_nifti(str(path)):
        return _extract_nifti_metadata(path)

    name = _lower_name(path)
    if not any(name.endswith(ext) for ext in NATURAL_IMAGE_EXTENSIONS):
        raise ValueError(
            f"Unsupported image format: {path.suffix}. "
            f"Supported formats: {', '.join(sorted(SUPPORTED_IMAGE_EXTENSIONS))}"
        )

    try:
        with Image.open(path) as image:
            width, height = image.size
            mode = image.mode
            image_format = image.format
    except Exception as exc:
        raise ValueError(
            f"Failed to read image metadata for {redact_path(str(path))}: {exc}"
        ) from exc

    return {
        "source_format": (image_format or "raster").lower(),
        "is_medical": False,
        "rows": height,
        "columns": width,
        "mode": mode,
        "modality_hint": None,
    }


def infer_modality_hint(
    modality_code: Optional[str] = None,
    body_part: Optional[str] = None,
) -> Optional[str]:
    """Map DICOM technical codes to a Medical AgenticIQA profile key."""
    code = (modality_code or "").strip().upper()
    part = (body_part or "").strip().upper()

    if code in _DICOM_MODALITY_TO_PROFILE:
        return _DICOM_MODALITY_TO_PROFILE[code]
    if part in _BODY_PART_TO_PROFILE:
        return _BODY_PART_TO_PROFILE[part]
    return None
