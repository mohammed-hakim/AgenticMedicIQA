"""Privacy helpers for logging and prompt construction.

These helpers are deliberately small and conservative. They reduce accidental
exposure of file paths and metadata in logs; they are not a compliance layer and
do not remove burned-in protected health information (PHI) from pixels.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

# Tags that must never be copied into prompts, API responses, or logs by this
# project. The medical ingestion code only reads an explicit allow-list of
# non-identifying attributes.
SENSITIVE_METADATA_KEYS = {
    "patientname",
    "patientid",
    "patientbirthdate",
    "patientsex",
    "patientage",
    "patientaddress",
    "patienttelephone",
    "accessionnumber",
    "studyid",
    "institutionname",
    "institutionaddress",
    "referringphysicianname",
    "performingphysicianname",
    "operatorsname",
    "deviceSerialNumber".lower(),
    "stationname",
}

_WINDOWS_PATH = re.compile(r"[A-Za-z]:\\[^\s\"'<>|]+")
_POSIX_PATH = re.compile(r"(?<![\w.])/(?:[\w.\-]+/)+[\w.\-]+")


def _suffix_label(path: str) -> str:
    name = Path(str(path)).name.lower()
    if name.endswith(".nii.gz"):
        return ".nii.gz"
    return Path(name).suffix


def redact_path(value: Optional[str]) -> str:
    """Return a log-safe placeholder for a file path.

    The placeholder keeps only the extension so operators can distinguish input
    types without exposing folder names or file names that may contain PHI.
    """
    if not value:
        return "<no-path>"
    suffix = _suffix_label(value)
    return f"<redacted-image>{suffix}" if suffix else "<redacted-path>"


def redact_text(value: str) -> str:
    """Redact absolute Windows and POSIX paths embedded in a text message."""
    if not value:
        return value
    redacted = _WINDOWS_PATH.sub("<redacted-path>", value)
    redacted = _POSIX_PATH.sub("<redacted-path>", redacted)
    return redacted


def redact_metadata(metadata: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Drop potentially identifying keys from a metadata dictionary."""
    if not metadata:
        return {}
    clean: Dict[str, Any] = {}
    for key, value in metadata.items():
        if str(key).strip().lower() in SENSITIVE_METADATA_KEYS:
            continue
        if isinstance(value, dict):
            clean[key] = redact_metadata(value)
        else:
            clean[key] = value
    return clean


def safe_metadata_subset(metadata: Optional[Dict[str, Any]], keys: Iterable[str]) -> Dict[str, Any]:
    """Return an allow-listed subset of metadata."""
    if not metadata:
        return {}
    return {key: metadata[key] for key in keys if key in metadata}
